import pretty_midi
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.neural_blocks import ( SkipConnMLP, FourierEncoder )
from src.dataset import MIDIDataset
from src.utils import mse2psnr

from tqdm import trange
from itertools import chain

resolution = 0.05

def round_to(val, rnd=resolution): return (val + rnd-0.001)//rnd

# torch.autograd.set_detect_anomaly(True)

device="cuda"
PAD = 128

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--dir", default="data/maestro-v3.0.0/2018")
  # TODO maybe make num samples stochastic?
  a.add_argument("--num-samples", type=int, default=5000)
  a.add_argument("--cutoff", type=int, default=2000)
  a.add_argument("--epochs", type=int, default=1000)

  a.add_argument("--out-samples", nargs="+", default=[500, 3000])
  a.add_argument("--song-latent", type=int, default=128)
  a.add_argument("--emb-size", type=int, default=128)
  a.add_argument("--with-vel", action="store_true")
  a.add_argument("--total-secs", type=int, default=60)

  return a.parse_args()

def main():
  args = arguments()
  BS = 5
  dataset = MIDIDataset(args.dir, args.num_samples)
  loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BS,
    shuffle=True,
  )

  latent = torch.randn(
    len(dataset),
    args.song_latent,
    requires_grad=True,
    device=device,
  )

  times = torch.linspace(0, 1, args.num_samples, device=device)[None,...,None]

  model = SkipConnMLP(
    in_size=1, out=128,
    latent_size=args.song_latent,
    hidden_size=512,
    enc=FourierEncoder(input_dims=1, sigma=1<<6),
  ).to(device)

  #params = chain(model.parameters(), latent)
  params = chain([latent], model.parameters())

  opt = torch.optim.Adam(params, lr=1e-3)#, weight_decay=0)
  t = trange(args.epochs)
  max_psnr = -1
  best_epoch = -1
  for e in t:
    for exp, i in loader:
      exp = exp.to(device)
      opt.zero_grad()
      got = model(
        times.expand(exp.shape[0], -1,-1),
        latent[i][:,None,:].expand(-1, exp.shape[1], -1)
      )
      loss = F.mse_loss(got, exp)
      #loss = F.binary_cross_entropy_with_logits(got, exp)

      psnr = mse2psnr(loss).item()
      max_psnr = max(max_psnr, psnr)
      if max_psnr == psnr: best_epoch = e
      t.set_postfix(L=f"{loss.item():.03e}", PSNR=f"{psnr:.01f}")
      loss.backward()
      opt.step()

  print(max_psnr, best_epoch)

  # create some sample outputs
  for s in args.out_samples:
    t = torch.linspace(0,1,s,device=device).unsqueeze(-1)
    # rand version
    l = torch.rand(1, args.song_latent, device=device).expand(s, -1)
    to_notes(model(t, l), s, args, f"outputs/rand1_{s}.mid")
    l = torch.rand(1, args.song_latent, device=device).expand(s, -1)
    to_notes(model(t, l), s, args, f"outputs/rand2_{s}.mid")
    l = torch.rand(1, args.song_latent, device=device).expand(s, -1)
    to_notes(model(t, l), s, args, f"outputs/rand3_{s}.mid")

    l = latent[None, 0].expand(s, -1)
    to_notes(model(t, l), args, f"outputs/recover_{s}.mid")

    #to_notes(model(t, latent), args, f"outputs/held_{s}.mid")


def to_notes(output, samples, args, name="example.mid"):
  sec = args.total_secs
  closest = []
  for p in output:
    p = p[:-1]
    # threshold those greater than 0.9
    p = (p > 0.75).nonzero()
    closest.append(p.tolist())


  cello_out = pretty_midi.PrettyMIDI()
  # Create an Instrument instance for a cello instrument
  cello = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))

  curr_end = 0
  dur = sec/samples
  def add_note(p):
    if p == PAD: return
    note = pretty_midi.Note(
      velocity=65,
      # Most things take 1-128, not 0-127?
      # For some reason without the odd clamping it doesn't work?
      pitch=min(max(int(p),7), 120),
      start=curr_end,
      end=curr_end+dur,
    )
    cello.notes.append(note)
  for p in closest:
    if isinstance(p, list):
      for sp in p:
        assert(len(sp) == 1)
        add_note(sp[0])
    else: add_note(p.item())
    curr_end += dur
  cello_out.instruments.append(cello)
  # Write out the MIDI data
  cello_out.write(name)

if __name__ == "__main__": main()

