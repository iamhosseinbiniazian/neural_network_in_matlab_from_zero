function Images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

Images = fread(fp, inf, 'unsigned char');
Images = reshape(Images, numCols, numRows, numImages);
Images = permute(Images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
Images = reshape(Images, size(Images, 1) * size(Images, 2), size(Images, 3));
% Convert to double and rescale to [0,1]
Images = double(Images) / 255;

end