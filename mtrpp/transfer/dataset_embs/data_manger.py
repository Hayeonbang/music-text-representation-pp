from torch.utils.data import DataLoader
from mtrpp.transfer.dataset_embs.annotation import Annotation_Dataset
# from mtrpp.transfer.dataset_embs.youtube import Youtube



def get_dataloader(args, split, audio_embs):
    dataset = get_dataset(
        eval_dataset= args.eval_dataset,
        data_path= args.msu_dir,
        split= split,
        audio_embs= audio_embs
    )
    if split == "TRAIN":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "VALID":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "TEST":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "ALL":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    return data_loader


def get_dataset(
        eval_dataset,
        data_path,
        split,
        audio_embs
    ):
    if eval_dataset == "annotation":
        dataset = Annotation_Dataset(data_path, split, audio_embs)
    elif eval_dataset == "youtube":
        dataset = Youtube_Dataset(data_path, split, audio_embs)

    else:
        print("error")
    return dataset