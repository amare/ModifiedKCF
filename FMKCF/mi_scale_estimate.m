function [ delta_offset,angle_factor,scale_factor ] = mi_scaleEstimate( obj,patch,params )
%MI_SCALEESTIMATE 
%   ���ø���Ҷ÷�ֱ任����Ŀ����ģ��͸�������֮���ƫ��delta_offset([x,y]), �Ƕȱ仯angle_factor, ��Գ߶�scale_factor.
%   objΪĿ��ģ�壬patchΪ���ٵ���Ŀ������paramsΪ��ز�����
%   ʱ��: 128x128�ߴ�Լ0.03��.

delta_offset=[0,0];

sidelen=2^(floor(log(max(size(obj,1),size(obj,2)))/log(2))); %�����λ��ı߳�
% if sidelen>32
%     sidelen=32;
% end
if size(obj,1)>size(obj,2) %�����߳�Ϊ�ο��߳������ಹ��    
    sz=size(obj);
    tmp=floor(sidelen*sz(2)/sz(1));
    img1=imresize(obj,[sidelen,tmp]);
    img2=imresize(patch,[sidelen,tmp]);
    xs = (1:sidelen) - floor((sidelen-tmp)/2);
    ys = (1:sidelen);
    xs(xs < 1) = 1;
    xs(xs > tmp) = tmp;
    img1 = img1(ys, xs, :);
    img2=img2(ys,xs,:);
else
    sz=size(obj);
    tmp=floor(sidelen*sz(1)/sz(2));
    img1=imresize(obj,[tmp,sidelen]);
    img2=imresize(patch,[tmp,sidelen]);
    xs = (1:sidelen);
    ys = (1:sidelen) - floor((sidelen-tmp)/2);
    ys(ys < 1) = 1;
    ys(ys > tmp) = tmp;
    img1 = img1(ys, xs, :);
    img2=img2(ys,xs,:);
end
inimg1=img1;
inimg2=img2;
params.window_type='cosine';
params.correlation_type='phase';
params.feature='gray';
[M,N]=size(img1);

%��ȡ����ͼ
if strcmp(params.feature,'gray')
    h=fspecial('gaussian',[3,3],1);
    img2=imfilter(img2,h);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    img1=inimg1-mean(inimg1(:));
    img2=img2-mean(img2(:));
elseif strcmp(params.feature,'graynorm')
    h=fspecial('gaussian',[3,3],1);
    img2=imfilter(img2,h);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    img1=inimg1-mean(inimg1(:));
    img1s=abs(img1);
    mean1=mean(img1s(:));
    img1(img1>mean1)=mean1;
    img1(img1<-mean1)=-mean1;
    img1=img1/(mean1+eps);
    img2=img2-mean(img2(:));  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    img2s=abs(img2);
    mean2=mean(img2s(:));
    img2(img2>1.5*mean2)=1.5*mean2;
    img2(img2<-1.5*mean2)=-1.5*mean2;
    img2=img2/(1.5*mean2+eps);
elseif strcmp(params.feature,'edgefilter')
    h2=fspecial('log',[5,5],0.5);
    %         h2=fspecial('log',[5,5],0.4);
    %         h2=fspecial('laplacian',0.5);
    %         h2=fspecial('motion');
    %         h2=fspecial('prewitt');
    %         h2=fspecial('sobel');
    img1=imfilter(inimg1,h2)*1;
    img2=imfilter(inimg2,h2)*1;
elseif strcmp(params.feature,'edgelog')
    img1=edge(inimg1,'log');
    img2=edge(inimg2,'log');
elseif strcmp(params.feature,'edgecanny')
    img1=edge(inimg1,'canny');
    img2=edge(inimg2,'canny');
end

%�Ӵ�
if strcmp(params.window_type,'cosine')
    cos_window = hann(size(img1,1)) * hann(size(img1,2))'; 
    img1=img1.*cos_window;
    img2=img2.*cos_window;
else
    band=floor(min(M,N)*0.1);
    mask1=zeros(size(img1,1),size(img1,2));
    mask1(band+1:end-band,band+1:end-band)=1;
    mask2=zeros(size(img2,1),size(img2,2));
    mask2(band+1:end-band,band+1:end-band)=1;
    img1=img1.*mask1;
    img2=img2.*mask2;
end
img1s=img1;
img2s=img2;

%����Ҷ÷�ֱ任?
if 1 %strcmp(correlationtype,'F-M')
    img1f=ifftshift(fft2(img1));    %תΪƵ��ͼ 
    img2f=ifftshift(fft2(img2));
    if 1 %(nums==1)
        H=img1;
        for i=1:M
            for j=1:N
                tmp=cos(3.1415*(i-0.5*M)/M)*cos(3.1415*(j-0.5*N)/N);
                H(i,j)=(1-tmp)*(2-tmp);
            end
        end
    end
    img1ff=abs(img1f).*H;   %��Ƶ��ͼ��Ȩ
    img2ff=abs(img2f).*H;
    img1lp=imlogpolar(img1ff,50,360,'bilinear');    %�Լ�����任
    img2lp=imlogpolar(img2ff,50,360,'bilinear');
    cos_window2 = hann(size(img1lp,1)) * hann(size(img1lp,2))'; 
    img1lp=img1lp.*cos_window2; %�Լ�����任��ľ���Ӵ�
    img2lp=img2lp.*cos_window2;
    img1lpf=fft2(img1lp);
    img2lpf=conj(fft2(img2lp));
    imglpf=img1lpf.*img2lpf./(abs(img1lpf).*abs(img2lpf)+eps);  %��λ��ط�?    
    imglp=ifftshift((ifft2(imglpf)));
    [anglerow,anglecol]=find(imglp==max(imglp(:)),1);   %���ֵ��Ӧ��ƫ��Ϊ�ǶȺͳ߶�ָ��
    scale_factor=(sidelen/2)^(anglerow/50-26/50);   %������Գ߶�
    angle_factor=anglecol-181;  %����Ƕȱ仯
    
    if true    %���ݽǶȺͳ߶ȱ仯��ͼ�����У��
        img1=imresize(img1,scale_factor);
        if scale_factor<1
            img1_tmp=zeros(sidelen,sidelen);
            img1_tmp(sidelen/2-floor(size(img1,1)/2)+(1:size(img1,1)),sidelen/2-floor(size(img1,2)/2)+(1:size(img1,2)))=img1;
            img1=img1_tmp;
        else
            img1=img1(floor(size(img1,1)/2-sidelen/2)+(1:sidelen),floor(size(img1,2)/2-sidelen/2)+(1:sidelen));
        end
        img1=imrotate(img1,anglecol-181,'bicubic','crop');
    end
end

if true     %��У���������ͼ��������ƫ����    
    img1f=fft2(img1);
    img2f=conj(fft2(img2));
    if strcmp(params.correlation_type,'phase')
        convf=img1f.*img2f./(abs(img1f).*abs(img2f)+eps); 
    else    %strcmp(params.correlation_type,'cross')
        convf=img1f.*img2f;
    end
    iconvf=ifftshift(real(ifft2(convf)));
    [row,col]=find(iconvf==max(iconvf(:)),1);
    delta_offset=[row,col]-floor(size(img1)/2)-[1,1];  
end

% angle_factor
% scale_factor
end
