% usage: octave-cli run_uciqe.m <in.mat> <out.txt>; in.mat holds variables im1..imN (uint8 HxWx3)
args = argv();
addpath('../uwrepos/UCIQE');
S = load(args{1}); n = numel(fieldnames(S));
out = fopen(args{2}, 'w');
for i = 1:n
  fprintf(out, '%.8f\n', UCIQE(S.(sprintf('im%d', i))));
end
fclose(out);
