clear;clc;

addpath('Codes');
setenv('LC_ALL', 'C')


sub_dataset='WildSketch'

if strcmp(sub_dataset,'CUHK')
    Start_Id = 1;
    Nim = 100;
    Database = 'CUFS';
end
if strcmp(sub_dataset,'AR')
    Start_Id = 101;
    Nim = 43;
    Database = 'CUFS';
end
if strcmp(sub_dataset,'XM2VTS')
    Start_Id = 144;
    Nim = 195;
    Database = 'CUFS';
end
if strcmp(sub_dataset,'CUFSF')
    Start_Id = 1;
    Nim = 944;
    Database = 'CUFSF';
end
if strcmp(sub_dataset,'SYSU')
    Start_Id = 1;
    Nim = 400;
    Database = 'WildSketch';
end

Exp = 'WildSketch_Experiment'
epoch=100

refSketchPath = ['../dataset/',Database,'/test/sketches/'];
synSketchPath = ['../results/',Exp,'/test_',int2str(epoch),'/images/'];

S_Ours_SSIM = zeros(Nim,1);
S_Ours_FSIM = zeros(Nim,1);
S_Ours_SCOOT = zeros(Nim,1);

for i = Start_Id:Start_Id+Nim-1

    if strcmp(sub_dataset,'WildSketch')
        refim = imread([refSketchPath,num2str(i),'.png']);
    else
        refim = imread([refSketchPath,num2str(i),'_0.jpg']);
    end
    if size(refim,3) == 3
        refim = rgb2gray(refim);
    end

    im = imread([synSketchPath, num2str(i), '_fake_B.png']);
    if size(im,3) == 3
        im = rgb2gray(im);
    end
    [h w ch] = size(refim);
    im = imresize(im,[h w],'bicubic');

    S_Ours_SCOOT(i-Start_Id+1) = ScootMeasure(refim, im);

    refim = double(refim);
    im = double(im);
    S_Ours_SSIM(i-Start_Id+1) = FR_SSIM(refim, im);
    S_Ours_FSIM(i-Start_Id+1) = FeatureSIM(refim, im);

end

fprintf('Sample %d, mean SSIM: %f\t mean FSIM: %f\t mean SCOOT: %f\n', ...   
        Nim, mean(S_Ours_SSIM), mean(S_Ours_FSIM), mean(S_Ours_SCOOT));

