function compile(cudnn_dir)
outdir = 'build';
include = '-I./include/';
if nargin < 1
    cudnn_dir = '/usr/local/cuda/lib64';
end
cudnn_dir = ['-L' cudnn_dir];

cd './mex'
try
    mexcuda('mex3DConv.cu', include, '-outdir', outdir, ...
        '-lstdc++', '-lc', '-lcudnn', cudnn_dir);
    mexcuda('mex3DConvt.cu', include, '-outdir', outdir, ...
        '-lstdc++', '-lc', '-lcudnn', cudnn_dir);
catch ME
    cd '../'
    error(ME.message)
end
cd '../'

end