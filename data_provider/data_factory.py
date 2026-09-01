from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

def data_provider(args, flag):
    """Build a validated dataset/DataLoader pair for one split."""
    if args.data != 'custom':
        raise ValueError("CE-BiD supports CSV data through --data custom only.")
    if flag not in {'train', 'val', 'test'}:
        raise ValueError(f"Unsupported data split '{flag}'.")
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    if args.num_workers < 0:
        raise ValueError('num_workers cannot be negative.')

    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = flag != 'test'
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    specific_start = getattr(args, 'specific_start', None)
    specific_end = getattr(args, 'specific_end', None)
    train_ratio = getattr(args, 'train_ratio', None)
    val_ratio = getattr(args, 'val_ratio', None)
    test_ratio = getattr(args, 'test_ratio', None)

    data_set = Dataset_Custom(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        step=args.step,
        total_steps=args.total_steps,
        specific_start=specific_start,
        specific_end=specific_end,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    if flag in {'train', 'val'} and len(data_set) == 0:
        raise ValueError(
            f"The {flag} split has no forecasting windows. Increase the data window "
            f"or reduce seq_len ({args.seq_len}) / pred_len ({args.pred_len})."
        )
    print(f'{flag}: {len(data_set)} windows')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
