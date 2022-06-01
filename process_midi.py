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
import random

#torch.autograd.set_detect_anomaly(True)

device="cuda"
PAD = 128

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--dir", default="data/maestro-v3.0.0/2011")
  # TODO maybe make num samples stochastic?
  a.add_argument("--num-samples", type=int, default=500)
  a.add_argument("--epochs", type=int, default=1000)

  a.add_argument("--out-samples", nargs="+", default=[100, 250, 500])
  a.add_argument("--song-latent", type=int, default=128)
  a.add_argument("--total-secs", type=int, default=60)

  return a.parse_args()

def main():
  args = arguments()
  BS = 100
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

  model = SkipConnMLP(
    in_size=1, out=128,
    latent_size=args.song_latent,
    hidden_size=512,
    enc=FourierEncoder(input_dims=1, sigma=1<<6),
  ).to(device)

  params = chain(model.parameters())

  opt = torch.optim.Adam(params, lr=5e-4)
  t = trange(args.epochs)
  max_psnr = -1
  best_epoch = -1
  for e in t:
    for exp, times, i in loader:
      exp = exp.to(device)
      times = times.to(device)

      opt.zero_grad()
      got = model(
        times[...,None].expand(exp.shape[0], -1,-1),
        latent[i][:,None,:].expand(-1, exp.shape[1], -1)
      )
      loss = F.mse_loss(got, exp.float())
      loss.backward()
      opt.step()

      #loss = F.binary_cross_entropy_with_logits(got, exp.float())

      psnr = mse2psnr(loss).item()
      max_psnr = max(max_psnr, psnr)
      if max_psnr == psnr: best_epoch = e
      t.set_postfix(L=f"{loss.item():.03e}", PSNR=f"{psnr:.01f}")

  print(max_psnr, best_epoch)

  # create some sample outputs
  for s in args.out_samples:
    t = torch.linspace(0,1,s,device=device).unsqueeze(-1)
    for n in range(10):
      l = torch.randn(1, args.song_latent, device=device).expand(s, -1)
      to_notes(model(t, l), s, args, f"outputs/rand{n}_{s}.mid")

    l = latent[None, 0].expand(s, -1)
    to_notes(model(t, l), s, args, f"outputs/recover_{s}.mid")

    l = latent[None, 0].expand(s, -1)
    l = l + torch.randn_like(l) * 1e-2
    to_notes(model(t, l), s, args, f"outputs/perturb_{s}.mid")

    l = latent[:3].mean(dim=0, keepdim=True).expand(s, -1)
    to_notes(model(t, l), s, args, f"outputs/mix_{s}.mid")


def to_notes(output, samples, args, name="example.mid"):
  sec = args.total_secs
  closest = []
  for p in output:
    p = p[:-1]
    p = (p > 0.5).nonzero()
    closest.append(p.tolist())


  cello_out = pretty_midi.PrettyMIDI()
  # Create an Instrument instance for a cello instrument
  cello = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))

  curr_end = 0
  dur = 0.5
  def add_note(p, count):
    if p == PAD: return 0
    note_dur = (count+1)*dur
    note = pretty_midi.Note(
      velocity=65,
      # Most things take 1-128, not 0-127?
      # For some reason without the odd clamping it doesn't work?
      pitch=min(max(int(p),10), 118),
      start=curr_end,
      end=curr_end+note_dur,
    )
    cello.notes.append(note)
    return note_dur
  def add_notes(ps, count):
    for sp in ps:
      assert(len(sp) == 1)
      add_note(sp[0], count)
    note_dur = (count + 1) * dur
    # shrink rests
    if len(ps) == 0: note_dur *= 2/3
    return note_dur

  # merging strategy: greedily merge if the notes are the same
  prev = None
  count = 0
  for p in closest:
    if p == prev:
      count += 1
      continue

    if isinstance(prev, list): add_len = add_notes(prev, count)
    elif prev is None: add_len = 0
    else: add_len = add_note(prev.item(), count)

    curr_end += add_len

    prev = p
    count = 0

  if isinstance(prev, list): add_notes(prev, count)
  elif prev is None: ...
  else: add_note(prev.item(), count)

  cello_out.instruments.append(cello)
  # Write out the MIDI data
  cello_out.write(name)

if __name__ == "__main__": main()

