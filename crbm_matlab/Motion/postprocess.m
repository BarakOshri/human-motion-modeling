% Version 1.01 
%
% Code provided by Graham Taylor, Geoff Hinton and Sam Roweis 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%
% This program calls the two post-processing 
% It is intended to be called after generating a sequence
% It assumes the generated sequence is in "visible"
% It writes the post-processed sequence to "newdata"
function newdata = postprocess(visible, data_std, data_mean, offsets)
skel = buildskel_mit();
%assemble the data back into the "body-centred" coordinates and scale back
newdata = postprocess1(visible, data_std, data_mean, skel, offsets);
%back into reference coordinate system
newdata = postprocess2(newdata, skel);
end