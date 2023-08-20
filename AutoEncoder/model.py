import torch
import torch.nn as nn

#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()): # check latent_dim
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # 16, (240, 240)
        act_fn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # 32, (120, 120)
        act_fn,
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # 64, (60, 60)
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(4*out_channels, 8*out_channels, 3, padding=1, stride=2), # 128, (30, 30)
        act_fn,
        nn.Conv2d(8*out_channels, 8*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(8*out_channels, 16*out_channels, 3, padding=1, stride=2), # 256, (15, 15)
        act_fn,
        nn.Conv2d(16*out_channels, 16*out_channels, 3, padding=1),
        act_fn,
        nn.Flatten(),
            # import pdb; pdb.set_trace()

        nn.Linear(16*out_channels*15*15, latent_dim),  #256*15*15    # check size of the kernel
        act_fn
    )

  def forward(self, x):
    x = x.view(-1, 3, 240, 240)
    output = self.net(x)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 16*out_channels*15*15),
        act_fn
    )
    # import pdb; pdb.set_trace()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(16*out_channels, 16*out_channels, 3, padding=1), # (15, 15)
        act_fn,
        nn.ConvTranspose2d(16*out_channels, 8*out_channels, 3, padding=1,
                           stride=2, output_padding=1),
        act_fn,
        nn.ConvTranspose2d(8*out_channels, 8*out_channels, 3, padding=1), # (30, 30)
        act_fn,
        nn.ConvTranspose2d(8*out_channels, 4*out_channels, 3, padding=1,
                           stride=2, output_padding=1), # (60, 60)
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1,
                           stride=2, output_padding=1), # (120, 120)
        act_fn,
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1,
                           stride=2, output_padding=1), # (240, 240)
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 16*self.out_channels, 15, 15)
    output = self.conv(output)
    return output


#  defining autoencoder
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded