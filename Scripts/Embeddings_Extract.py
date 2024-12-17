

import torch
import shutil
from pathlib import Path
from esm import FastaBatchedDataset, pretrained, MSATransformer

MAX_NUM_EMBS_PER_FILE = 10000

def get_labels(sequence_labels) -> list:
    def format_label(label: str):
        label = label.split('|')
        organism = label[0].split('-')[0]
        mutation = label[1]
        return organism + '_' + mutation
    def remove_comment(label: str):
        idx = label.rfind('|')
        if ' ' in label[idx:]:
            idx = label[idx:].index(' ') + idx 
            label = label[:idx]   
        return label

    if isinstance(sequence_labels, str):
        return format_label(remove_comment(sequence_labels)) 

    for i, label in enumerate(sequence_labels):
        sequence_labels[i] = format_label(remove_comment(label))

    return sequence_labels

def extract(model_location, fasta_file, output_file, toks_per_batch=4066, truncation_seq_length=1022, nogpu=False, warm_start=True):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    labels = get_labels(dataset.sequence_labels)
    dataset.sequence_labels = labels

    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches)
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    with torch.no_grad():
        tmp_batches_d = Path(output_file).parent.joinpath('tmp_embs_extract')
        tmp_batches_d.mkdir(exist_ok=warm_start)
        tmp_batches_f = [f.name for f in tmp_batches_d.iterdir() if f.is_file()]

        all_batch_file = []
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")

            batch_file = f'batch_{batch_idx}.pt'
            if batch_file in tmp_batches_f:
                print(f'batch {batch_idx} already processed')
                all_batch_file.append(Path(tmp_batches_d).joinpath(batch_file))
                continue

            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=model.num_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            result = {}
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                result[label] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }            

            all_batch_file.append(Path(tmp_batches_d).joinpath(batch_file))
            torch.save(
                result,
                all_batch_file[-1]
            )

        final_results = {}
        file_count = 0
        out_path = Path(output_file).with_suffix('')
        max_len = MAX_NUM_EMBS_PER_FILE - len(torch.load(all_batch_file[0]))

        print(f'gathering embeddings into:\n- {Path(out_path)}\n')

        all_saved = True
        for i, batch_file in enumerate(all_batch_file):
            current_result = torch.load(batch_file)
            final_results.update(current_result)
            
            if len(final_results) >= max_len or i == len(all_batch_file) - 1:
                file_to_save = str(out_path) + f"_{file_count}" + ".pt"
                torch.save(
                    final_results,
                    file_to_save,
                )
                file_count += 1
                final_results = {}
                
                save_msg = f'> {Path(file_to_save).name}'
                if Path(file_to_save).is_file() == False:
                    save_msg += 'failed'
                    all_saved = False
                else:
                    save_msg += 'saved'
                print(save_msg)

            if all_saved:
                shutil.rmtree(tmp_batches_d)
            else:
                print('embeddings gathering failed')
                print(f'individual embeddings available in:\n- {Path(tmp_batches_d)}\n')

