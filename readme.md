# Neural Music Interpolation (NeMuI) 😪

This is a fun project for reconstructing music I made in a bit of downtime.

## Quickstart

```
git clone git@github.com:JulianKnodt/nemui.git
cd nemui
python3 process_midi.py --dir <directory with midi files>
```

## Objective

I've always been interested in generating music, but was never quite sure how to go about doing
it, but I finally tried messing at it with for a little bit, and I think my understanding of
what "generating music" is has been expanded a bit. For those who are interested in creating
their own music generator, I'll outline what I learned from this project and how it works a bit.

### What is "music"?

A strange question to ask, but what is "music"? Broadly speaking, music is hard to define, but
is something that can be thought as being nice to listen to. But when I refer to generating
"music", that definition is too broad. In general, you may think of the most fundamental form of
music as the raw [waveform](https://en.wikipedia.org/wiki/Waveform), which corresponds directly
to the sound heard. Any sound we can hear would be representable as a waveform, but it
is not very amenable to being generated, because there are an extremely high number of samples,
making it difficult to process. For example, most `.wav` files will encode 44100 samples per
second, so for a minute of music it would be 200k samples for a single song. Thus, it is
easier to operate on a more compressed representation of music. For this purpose, I use scores,
or at least the digital version. Scores encode a quantized version of possible frequencies, over
much coarser time scales, making them useful for roughly describing music in a rapdily readable
form. As opposed to a raw waveform though, there is a lot of room for interpretation, as a
single note can be played in many different ways, but this isn't so important when
generating music. Thus, for my purposes, I am not generating "music", but
generating scores, due to ease of representation.

This is actually a pretty important limitation, as most songs will only have the raw waveform.
It's not possible yet to go from the raw waveform to a midi.

### How to generate scores?

Now that there is a representation, how do are scores created/generated? For this, since I
am not very musically inclined, the thing that makes the most sense to would be to take existing
music, and somehow blend them to generate new music. To do that, I chose to represent MIDI
scores as the output of some function `fn(song) -> song notes`, and attempt to reconstruct
multiple songs using the same function. Then it's possible to do something like
`fn(0.5 * (song1 + song))`, and generate some blend of the songs. Now the question is how to
actually encode each song and set of notes as a something to be optimized over? For each song,
I take a simple approach, just setting it as a random, fixed-size vector.

Encoding the notes into an optimizable form is less clear. We would like to be able to represent
individual notes, as well as chords (multiple notes played simultaneously), at varying lengths.
First, I chose to handle the issue of varying lengths in a naive way: simply quantize at a coarse
granularity and if a note is played during some portion of that time it is treated as being
played on during that entire segment. The note representation is less clear. Ideally, it would
be good to use one number to represent all notes. But this prevents representing chords, so it's
necessary to have multiple values. To do that, notes are represented as a 128 length 0/1 vector,
from 0 to 127 (the values allowed by MIDI files), set to 1 if the note is on.
This representation is very sparse (and thus inefficient), but allows for encoding chords and
single notes, as well as rests (no notes). Because it's flexible, I chose this representation,
but certainly a more compact representation exists.

Oh, and finally, the function from input to output is multiple matrix multiplies, with
[LeakyReLu](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) functions in
between. In addition, the input to the function has a series of timestamps for the notes we
want, in the range 0 to 1 for each song. So to recover the first note, it is `fn(song,
0)`, or the middle note is `fn(song, 0.5)`. We encode timestamps using [Positional
Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/). What this
encoding does is it maps different timestamps to the same value, since it is essentially
`sin(wt)` for multiple values of w. Thus, distance parts of a song may be correlated by this
encoding, making it easier to reconstruct.

With this, we have a full function to compute notes from songs. But we still actually have to
run this on some music in order to be able to generate new scores.

### Optimizing/Generating scores

I tried out generating music using [Google's Maestro
Dataset](https://magenta.tensorflow.org/datasets/maestro). While the system works to accurately
reconstruct the input pieces for a small number of notes, reconstructing a whole song was
difficult, possibly because regardless of the song length, it was compressed into a size of 0 to 1. Interpolating between songs also was very disappointing, and somehow usually lead to scores
with a lot of rests. One thing that did work well was adding slight bits of noise to the random
vectors for each song, which kept it mostly the same but added slight variations.

Providing newly generated random vectors as songs also lead to mixed results, which I think is
fine, since it's trained on a relatively small dataset. It's unclear to me whether adding in more
data would make it work better, but it might lead to more interesting results. I also get
relatively impatient, and terminate optimization while it's still learning, so it may be
possible that using a bigger neural net with more data would lead better sounding results.

I didn't use any kind of quality metric, but I just listened to it by hand. I've also realized
that it doesn't matter if most of the generated music is crap, as long as a few are good then
the generator is interesting.


