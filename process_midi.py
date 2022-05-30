import pretty_midi
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.neural_blocks import ( SkipConnMLP, FourierEncoder )
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
  a.add_argument("--file", required=True)
  a.add_argument("--file2", required=True)
  # TODO maybe make num samples stochastic?
  a.add_argument("--num-samples", type=int, default=1000)
  a.add_argument("--cutoff", type=int, default=2000)
  a.add_argument("--epochs", type=int, default=5000)
  a.add_argument("--out-samples", nargs="+", default=[500, 3000])
  a.add_argument("--emb-size", type=int, default=128)
  a.add_argument("--with-vel", action="store_true")
  a.add_argument("--total-secs", type=int, default=60)
  a.add_argument("--emb-kind", choices=["emb", "1hot"], default="1hot")

  return a.parse_args()

def fat_sigmoid(x, eps=1e-3, training=True):
  y = x.sigmoid() * (1+2 * eps) - eps
  return y if training else y.clamp_(min=0, max=1)

def notes_zero(notes):
  return (notes == 128).all(dim=0).sum().clamp(min=0,max=1)

def main():
  args = arguments()

  if args.emb_kind == "1hot":
    emb = lambda x: F.one_hot(x, num_classes=129)[..., :-1]
  elif args.emb_kind == "emb":
    emb = nn.Embedding(
      # 128 + 1 for padding
      128+1,
      args.emb_size,
      device=device,
      padding_idx=PAD,
    )
  else: raise NotImplementedError()

  interpolate = lambda x: F.interpolate(
    x[None, None],
    size=(args.num_samples, x.shape[-1]),
    # TODO mess with mode?
    mode="bilinear",
  )[0,0]

  exp1_notes = preproc_midi(pretty_midi.PrettyMIDI(args.file)).to(device)
  exp2_notes = preproc_midi(pretty_midi.PrettyMIDI(args.file2)).to(device)

  def make_exp():
    return torch.stack([
      interpolate(emb(exp1_notes).sum(dim=0).float()),
      interpolate(emb(exp2_notes).sum(dim=0).float()),
    ], dim=0)

  times = torch.linspace(0, 1, args.num_samples, device=device)[None,...,None]\
    .expand(2,-1,-1)
  latent = torch.eye(2, device=device).unsqueeze(1)
  #latent = torch.rand_like(latent)
  latent = latent.expand(-1, args.num_samples, -1)

  if args.emb_kind == "1hot": out = 128
  elif args.emb_kind == "emb": out = args.emb_size

  model = SkipConnMLP(
    in_size=1, out=out,
    latent_size=2,
    hidden_size=512,
    enc=FourierEncoder(input_dims=1, sigma=1<<6),
  ).to(device)

  params = model.parameters()
  if hasattr(emb, "parameters"): params = chain(params, emb.parameters())
  opt = torch.optim.Adam(params, lr=1e-3)#, weight_decay=0)
  t = trange(args.epochs)
  max_psnr = -1
  best_epoch = -1
  for i in t:
    opt.zero_grad()
    exp = make_exp()
    got = model(times, latent)
    loss = F.mse_loss(got, exp)
    psnr = mse2psnr(loss).item()
    max_psnr = max(max_psnr, psnr)
    if max_psnr == psnr: best_epoch = i
    t.set_postfix(L=f"{loss.item():.03e}", PSNR=f"{psnr:.01f}")
    loss.backward()
    opt.step()

  print(max_psnr, best_epoch)

  # create some sample outputs
  for s in args.out_samples:
    t = torch.linspace(0,1,s,device=device).unsqueeze(-1)
    # rand version
    latent = torch.rand(s, 2, device=device)
    to_notes(model(t, latent), emb, args, f"rand_{s}.mid")

    # uniform lerp version
    latent = torch.stack([
      torch.linspace(0,1,s, device=device),
      torch.linspace(1,0,s, device=device),
    ], dim=-1)
    to_notes(model(t, latent), emb, args, f"lerp_{s}.mid")

    # hold one at one
    latent = torch.stack([
      torch.linspace(1,1,s, device=device),
      torch.linspace(0,1,s, device=device),
    ], dim=-1)
    to_notes(model(t, latent), emb, args, f"held_{s}.mid")


def to_notes(output, emb, args, name="example.mid"):

  sec = args.total_secs
  if args.emb_kind == "emb":
    dists = (emb.weight[None] * output[:,None,:]).sum(dim=-1)
    closest = dists.argmax(dim=-1)
  elif args.emb_kind == "1hot":
    closest = []
    for p in output:
      p = p[:-1]
      # threshold those greater than 0.9
      p = (p > 0.9).nonzero()
      closest.append(p.tolist())


  cello_out = pretty_midi.PrettyMIDI()
  # Create an Instrument instance for a cello instrument
  cello = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))

  curr_end = 0
  dur = sec/output.shape[0]
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



# preprocess midi file so that embedding later is easier
def preproc_midi(midi_data):
  curr = None
  # compute max number of samples for all instruments
  end = -1
  for instrument in midi_data.instruments:
    for note in instrument.notes: end = max(note.end, end)
  samples = int(round_to(end))

  for instrument in midi_data.instruments:
    if instrument.is_drum: continue
    pitches = [torch.full([samples], PAD)]

    for note in instrument.notes:
      start = int(round_to(note.start))
      end = int(round_to(note.end))

      assert(note.pitch < 128)
      i = 0
      while i < len(pitches) and (pitches[i][start:end] != PAD).any(): i += 1
      if i == len(pitches): pitches.append(torch.full([samples], PAD))
      # do not normalize, so we can subtract them out later
      pitches[i][start:end] = note.pitch

    # TODO encode each instrument separately
    #break
    add = torch.stack(pitches, dim=0)
    curr = add if curr is None else torch.cat([curr, add], dim=0)
  return curr

if __name__ == "__main__": main()
