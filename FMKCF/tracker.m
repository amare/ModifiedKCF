function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   GPS, 2017


	%if the target is large, lower the resolution, we don't need that much
	%detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';
    
    % 灰度特征的余弦窗
    y_gray = gaussian_shaped_labels(output_sigma, floor(window_sz));
    cos_window_gray = hann(size(y_gray,1)) * hann(size(y_gray,2))';
	
	
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    
    %尺度初始化
    scale.cur = 1.0;                %当前目标尺度
    scale.pre = 1.0;                %之前的目标尺度
    scale.tmp = 1.0;                %目标的临时尺度变量
    target_confidence = 1.0;        %目标置信度，即响应图中的最大响应值
    miss_threhold = 0.2;            %目标跟踪成败的参考阈值
    learning_rate = 0.075;          %模型更新学习率参数

	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end

		tic()

        % 更新阶段
		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
			switch kernel.type
			case 'gaussian',
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			case 'polynomial',
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear',
				kzf = linear_correlation(zf, model_xf);
			end
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection

			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            
            
            % 重新计算目标置信度，由跟踪到的目标位置为中心，计算得更准确一些
            % 第二个更新阶段
            patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
			switch kernel.type
			case 'gaussian',
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			case 'polynomial',
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear',
				kzf = linear_correlation(zf, model_xf);
            end
            
            newresponse = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            [new_vert_delta, new_horiz_delta] = find(newresponse == max(newresponse(:)), 1);
            target_confidence = max(newresponse(:));    
        end
        
        %目标跟踪成功与否的判定和学习率的自适应调整?    
        if target_confidence < miss_threhold
            new_learning_rate = 0;
            obj_missing = true;
        else
            new_learning_rate = learning_rate * (miss_threhold + (1 - miss_threhold) ./ (1 + exp(-15 * target_confidence + 7.5)));
            obj_missing = false;
        end
        
        params_tmp.feature = 'gray';
        patch_gray = mi_get_subwindow(im, pos, window_sz, params_tmp);     % 可能能够同 训练阶段的 patch = get_subwindow(im, pos, window_sz); 合并
        %估计目标新尺度
        if frame > 1 && ~obj_missing
            z_gray = bsxfun(@times, patch_gray, cos_window_gray); % 当前帧跟踪到的目标框灰度加窗特征(是否可以不加窗)
            x_gray = bsxfun(@times, model_x_gray, cos_window_gray); % 目标表观模板灰度加窗特征(是否可以不加窗)
            
            %剪切模板区域和目标框区域的大小为目标的2倍
            z_gray_tmp = z_gray(floor(target_sz(1)/2):(window_sz(1)-floor(target_sz(1)/2)-1),floor(target_sz(2)/2):(window_sz(2)-floor(target_sz(2)/2)-1),:);    
            x_gray_tmp = x_gray(floor(target_sz(1)/2):(window_sz(1)-floor(target_sz(1)/2)-1),floor(target_sz(2)/2):(window_sz(2)-floor(target_sz(2)/2)-1),:);
         
            [delta_offset,angle_factor,scale_factor]=mi_scale_estimate(z_gray_tmp, x_gray_tmp);    %目标位置偏移量、角度变化、尺度估计
            
            %为降低估计误差，旋转角度大的视为匹配错误  
            if abs(angle_factor)<5
                scale.tmp = scale.cur * scale_factor;
            else
                scale.tmp = scale.cur;
            end
            
            %disp(['scale temp:' num2str(scale.tmp)]);
            
            %当检测到的尺度变化较大时，用新尺度验证
            if abs(1 - scale_factor) > 0.05
                params_tmp.scale = scale.tmp;
                params_tmp.feature = 'hog';
                params_tmp.hog_orientations = 9;
                z_tmp = mi_get_subwindow(im, pos, window_sz, params_tmp); %在mi_get_subwindow函数中，把经过尺度缩放的窗口重新resize为当前窗口
                
                if ~isempty(cos_window),
                    zf = fft2(bsxfun(@times, z_tmp, cos_window));
                end
                
                %calculate response of the classifier at all shifts
                switch kernel.type
                    case 'gaussian',
                        kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                    case 'polynomial',
                        kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                    case 'linear',
                        kzf = linear_correlation(zf, model_xf);
                end
            
                response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
                
                % 当新尺度下的目标置信度小于原尺度，取原尺度
                if(max(response(:)) < miss_threhold || max(response(:)) < target_confidence)
                    scale.tmp = scale.cur;
                end
            end
            
            scale.pre = scale.cur;
            scale.cur = scale.tmp;
            
        end

        % 训练阶段
		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
		xf = fft2(get_features(patch, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
            model_x_gray = patch_gray;
		else
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
            model_x_gray = (1 - new_learning_rate) * model_x_gray + new_learning_rate * patch_gray;
        end

        %当尺度变化较大时，需要改变模板尺寸，并对模型整体进行更新
        %disp(['current scale: ' num2str(scale.cur)]);
        if abs(1 - scale.cur) > 0.1 && abs(1 - scale.cur) * min(target_sz(:)) > 1
            disp(['第' num2str(frame) '帧更新目标尺度']);
            disp(['current scale: ' num2str(scale.cur)]);
            target_sz = floor(target_sz * scale.cur);   %新的目标尺寸
            window_sz = floor(target_sz * (1 + padding));    %新的目标窗口大小
            
            % 新尺度下，更新标准置信图
            output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
            yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
            
            % 新尺度下，更新余弦窗
            cos_window = hann(size(yf,1)) * hann(size(yf,2))';
            
            y_gray = gaussian_shaped_labels(output_sigma, floor(window_sz));
            cos_window_gray = hann(size(y_gray,1)) * hann(size(y_gray,2))';
            
            % 新尺度下，更新目标表观模板
            patch = get_subwindow(im, pos, window_sz);
            model_xf = fft2(get_features(patch, features, cell_size, cos_window));
            
            % 新尺度下，更新目标灰度特征表观模板
            params_tmp.feature = 'gray';
            model_x_gray = mi_get_subwindow(im, pos, window_sz, params_tmp);
            
            % 新尺度下，更新回归系数矩阵
            %model_alphaf = 1 / (scale.cur ^ 2) * fft2(imresize((ifft2(model_alphaf)), window_sz / cell_size));
            
            %Kernel Ridge Regression, calculate alphas (in Fourier domain)
            switch kernel.type
            case 'gaussian',
                kf = gaussian_correlation(model_xf, model_xf, kernel.sigma);
            case 'polynomial',
                kf = polynomial_correlation(model_xf, model_xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kf = linear_correlation(model_xf, model_xf);
            end
            model_alphaf = yf ./ (kf + lambda);   %equation for fast training
            
            % 重置目标的相对尺度
            scale.cur = 1.0;
        end
        
		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();

		%visualization
		if show_visualization,
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end

