import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_b16_3D_config():
    """Returns the ViT-B/16 configuration."""
    # The simplest TransVNet for the 3D image segmentation 
    # No feature extraction via additional CNN encoder and no skip connections to the decoder CUP
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'

    config.decoder_channels = (128, 64, 32)
    config.skip_channels = (0, 0, 0) # For the decoder blocks associated with 1/16, 1/8, 1/4, 1/2, and 1/1 of input image size. Put zero if no skip connection is desired at a block!
    config.n_classes = 2
    return config


def get_conv_b16_3D_config():
    """Returns the Conv + ViT-B/16 configuration."""
    # The suggested TransVNet for the 3D image segmentation 
    # Feature extraction via additional CNN encoder and skip connections from the CNN encoder to the decoder CUP
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8, 8)})
    config.patches.grid = (5, 5, 5)
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'

    config.encoder_channels = (3, 8, 16, 32) #(3, 8, 16) # For the encoder blocks associated with 1/1, 1/2, and 1/4 of input image size
    # config.number_down_scaled = 2 # The number of half down scaling after the last encoder block by nn.MaxPool3D(2) without any convolutions
    config.decoder_channels = (128, 32, 16, 8) #(128, 32, 16) #(128, 32, 16, 8, 3) # First one is the first number of decoder input channels or head_channels
    config.skip_channels = (32, 16, 8, 3) #(16, 8, 3) #(16, 16, 16, 8, 3) # For the decoder blocks associated with 1/16, 1/8, 1/4, 1/2, and 1/1 of input image size. Put zero if no skip connection is desired at a block!
    config.n_classes = 2 # Number of classes for the image segmentation
    return config


def get_conv_b16_3D_gen_config():
    """Returns the Conv + ViT-B/16 configuration."""
    # The TransVNet for the 3D image generation (e.g., for material design purposes)
    # Feature extraction via additional CNN encoder and NO skip connections from the CNN encoder to the decoder CUP
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8, 8)})
    config.patches.grid = (5, 5, 5)
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'gen'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'

    config.encoder_channels = (3, 8, 16) # For the encoder blocks associated with 1/1, 1/2, and 1/4 of input image size
    # config.number_down_scaled = 2 # The number of half down scaling after the last encoder block by nn.MaxPool3D(2) without any convolutions
    config.label_size = 11 # The number of latent variables, which, in material design case study, is equal to the dimension of the labels or material properties
    config.decoder_channels = (128, 32, 16, 8, 3) # First one is the first number of decoder input channels or head_channels
    config.skip_channels = (0, 0, 0, 0, 0) # For the decoder blocks associated with 1/16, 1/8, 1/4, 1/2, and 1/1 of input image size. Put zero if no skip connection is desired at a block!
    config.n_classes = 2 # Number of classes for the image segmentation
    return config


def get_conv_b16_3D_gen2_config():
    """Returns the Conv + ViT-B/16 configuration."""
    # The suggested TransVNet for the 3D image generation (e.g., for material design purposes)
    # Feature extraction via additional CNN encoder and NO skip connections from the CNN encoder to the decoder CUP
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4, 4)})
    config.patches.grid = (4, 4, 4)
    config.hidden_size = 64 #768 #96 # Should be divisible by num_heads!
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 256 #3072 #384
    config.transformer.num_heads = 4 #12 #3
    config.transformer.num_layers = 2 #12 #3
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'gen'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'

    config.encoder_channels = (3, 8, 16, 64) # For the encoder blocks associated with 1/1, 1/2, and 1/4 of input image size
    # config.number_down_scaled = 2 # The number of half down scaling after the last encoder block by nn.MaxPool3D(2) without any convolutions
    config.label_size = 11 # The number of latent variables, which, in material design case study, is equal to the dimension of the labels or material properties
    config.decoder_channels = (64, 32, 16) # First one is the first number of decoder input channels or head_channels
    config.skip_channels = (0, 0, 0) #(16, 8, 3) # For the decoder blocks associated with 1/16, 1/8, 1/4, 1/2, and 1/1 of input image size. Put zero if no skip connection is desired at a block!
    config.n_decoder_CUPs = 7 # Number of classes for the image segmentation
    config.output_nonlinearity = 'sigmoid' # Nonlinearity in the last layer of the network; False meaning that there is no nonlineaity.
    config.n_classes = 1 # Number of classes for the image segmentation; 1 meaning that the output is a label image.
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config
