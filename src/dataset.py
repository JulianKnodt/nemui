from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import pretty_midi
import random

class MIDIDataset(Dataset):
  def __init__(self, dir, num_samples:int):
    self.midi_files = [
      os.path.join(dir, f) for f in os.listdir(dir)
      if f.lower().endswith(".mid") or f.lower().endswith(".midi")
    ][:]#this limit should be a parameter?
    self.midi_notes = [
      preproc_midi(pretty_midi.PrettyMIDI(midi))[1000:2500]
      for midi in self.midi_files
    ]
    self.midi_notes = [mn for mn in self.midi_notes if mn.shape[0] > num_samples]
    self.num_samples = num_samples

  def __len__(self): return len(self.midi_notes)
  def interpolate(self, x):
    beats = x.shape[0]
    start = random.randint(0,beats-self.num_samples)
    return x[start:start+self.num_samples],\
      torch.linspace(start/beats,(start+self.num_samples)/beats, self.num_samples, dtype=torch.float)
  def __getitem__(self, i):
    x, t = self.interpolate(self.midi_notes[i])
    return x, t, i

PAD = 128
resolution = 0.05
def round_to(val, rnd=resolution): return (val + rnd-0.001)//rnd

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

    add = torch.stack(pitches, dim=0)
    curr = add if curr is None else torch.cat([curr, add], dim=0)
  out = F.one_hot(curr, num_classes=129)[..., :-1].sum(dim=0).clamp(min=0, max=1)
  # compress empty space (maybe not the best idea)
  #i = 0
  #while i < out.shape[0]:
  #  if (out[i:i+8] == 0).all():
  #    out = torch.cat([out[:i], out[i+8:]],dim=0)
  #    continue
  #  i += 1
  return out
