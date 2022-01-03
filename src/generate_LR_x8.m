clear;close all;
setenv('LC_ALL','C');

%% settings
folder = '/data/SR/Flickr2K/';

scale_8 = 8; 

HR_folder = [folder, 'Flickr2K_HR/'];
LR_folder = [folder, 'Flickr2K_LR_bicubic/'];
mkdir(LR_folder);
mkdir([LR_folder, 'X8/']);
%% generate data
filepaths_bmp = dir(fullfile(HR_folder,'*.bmp'));
filepaths_jpg = dir(fullfile(HR_folder,'*.jpg'));
filepaths_png = dir(fullfile(HR_folder,'*.png'));
filepaths = [filepaths_bmp; filepaths_jpg; filepaths_png];   

for i = 1 : length(filepaths)
    disp(['i = ' num2str(i)]);
    disp(filepaths(i).name);
    file_name = strsplit(filepaths(i).name, '.');
    file_name_prefix = file_name{1};
    image_orig = imread(fullfile(HR_folder,filepaths(i).name));
    [h, w, c] = size(image_orig);
    display([h, w, c])
    
    % scale 8
    im_label_8 = modcrop(image_orig, scale_8);
    im_input_8 = imresize(im_label_8,1/scale_8,'bicubic');
    imwrite(im_input_8, [LR_folder, 'X8/', file_name_prefix, 'x8.png']);
end
