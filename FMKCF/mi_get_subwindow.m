function [ out ] = mi_getSubwindow( inimg, pos, winsz, params )
%GET_SUBWINDOW 
%   Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image inimg centered at pos ([y, x] coordinates),
%   with size winsz ([height, width]) and controlling params. If any pixels are outside of the image,
%   they will replicate the values at the borders.

%   Add supports for object scale and feature type chosen.
imgsz = [0, 0, 1];
imgsz(2) = size(inimg, 2);
imgsz(1) = size(inimg, 1);
if(size(inimg, 3) ~= 1)
    imgsz(3) = size(inimg, 3);
end

%imgsz=size(inimg);
newsz=winsz;
if ~isfield(params,'scale')
    params.scale = 1;
elseif params.scale~=1
    newsz=ceil(winsz*params.scale); %目标的相对尺度不为1时，调整剪切的区域大小
end

xs = floor(pos(2)) + (1:newsz(2)) - floor(newsz(2)/2);
ys = floor(pos(1)) + (1:newsz(1)) - floor(newsz(1)/2);

%% how to fill the pixels outside the image
% tmp_mean=mean(mean(inimg(ys(ys>1&ys<imgsz(1)),xs(xs>1&xs<imgsz(2)),:),1),2)+20;  
% im=cat(3,tmp_mean(1)*ones(winsz),tmp_mean(2)*ones(winsz),tmp_mean(3)*ones(winsz));
% im((ys>1&ys<imgsz(1)),(xs>1&xs<imgsz(2)),:)=inimg(ys(ys>1&ys<imgsz(1)),xs(xs>1&xs<imgsz(2)),:);
% im=uint8(im);

%检查超出边界的部分，并用边界像素值填充
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > imgsz(2)) = imgsz(2);
ys(ys > imgsz(1)) = imgsz(1);
im = inimg(ys, xs, :);

if params.scale~=1
    im=imresize(im,winsz);
end

%提取特征图
if nargin<4
    out = im;
else
    switch params.feature
        case 'gray'
            if(imgsz(3)>1)
                im=rgb2gray(im);
            end
            out=double(im)/255;
%             out=double(im)/255-0.5;
            tmp_mean=mean(out(:));
%             if ~isfield(params,'bg_mean')
%                 params.bg_mean=tmp_mean;
%             else
%                 params.bg_mean=params.bg_mean*0.7+0.3*tmp_mean;
%             end
            out=out-tmp_mean;
            
        case 'hog'
            orientations=9;
            if(~isempty(params.hog_orientations))
                orientations=params.hog_orientations;
            end
            if(imgsz(3)>1)
                im=rgb2gray(im);
            end
            out = double(fhog(single(im) / 255, 4, orientations));
            out(:,:,end)=[];
        case 'cn'
            if (~isfield(params,'w2cn') || isempty(params.w2cn))
                load('w2crs.mat');
                w2cn=w2crs;
            else
                w2cn=params.w2cn;
            end
            out=rgb2cn(double(uint8(im)),w2cn,-2);
%             for i=1:10
%                 outtmp=out(:,:,i);
%                 out(:,:,i)=out(:,:,i)-mean(outtmp(:));
%             end
            
        case 'edge'
            if(imgsz(3)>1)
                im=rgb2gray(im);
            end
            out=double(im)/255;
%             h=fspecial('gaussian',[7,7]);
%             out=imfilter(out,h);
            out=edge(out,'log');
        case 'dct'
            if(imgsz(3)>1)
                im=rgb2gray(im);
            end
            dctMat=dct2(double(im));
            out=dctMat;
        case 'rgb'
            out=im;
%             out(:,:,1)=out(:,:,1)-mean(mean(out(:,:,1)));
%             out(:,:,2)=out(:,:,2)-mean(mean(out(:,:,2)));
%             out(:,:,3)=out(:,:,3)-mean(mean(out(:,:,3)));
        case 'lab'
            cform=makecform('srgb2lab');
            out=applycform(im,cform);
            out=double(out)/255;
        case 'xyz'
            cform=makecform('srgb2xyz');
            out=applycform(im,cform);
            out=double(out)/255;
        case 'cmyk'
            cform=makecform('srgb2cmyk');
            out=applycform(im,cform);
            out=double(out)/255;
        otherwise
            out=im;
            
    end
end




end

