def loading_data():
  # Prepare data transformations and then combine them sequentially
  mean_std = cfg.DATA.MEAN_STD
  train_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip()
  ])
  val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
  ])
  img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
  ])
  target_transform = standard_transforms.Compose([
        own_transforms.MaskToTensor(),
        own_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
  ])
  restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
  ])
  # Load data
  full_training_data = resortit('train', simul_transform=train_simul_transform, transform=img_transform,
                           target_transform=target_transform)
  test_data = resortit('val', simul_transform=val_simul_transform, transform=img_transform,
                         target_transform=target_transform)
  # Create train and validation splits
  num_samples = len(full_training_data)
  training_samples = int(num_samples*0.8)
  validation_samples = num_samples - training_samples
  training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples,
  validation_samples])
  # Initialize dataloaders
  train_loader = torch.utils.data.DataLoader(training_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
  val_loader = torch.utils.data.DataLoader(validation_data, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False)
  
  return train_loader, val_loader, test_loader