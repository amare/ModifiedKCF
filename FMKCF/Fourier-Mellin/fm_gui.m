close all
global FM_PATH
colo=[0.9 0.9 0.9];

FM_PATH = [pwd filesep]  ;
disp('Change directory to the location of fm_gui.m...')
disp('...or change the value of FM_PATH in fm_gui.m.')

% ------------- the MAIN figure window ----------------------
h0 = figure('Units','normalized', ...
	'Color',colo, ...
	'CreateFcn','fm_guifun create', ...
	'Name','Fourier-Mellin Transform GUI', ...
	'NumberTitle','off', ...
	'PaperOrientation','landscape', ...
	'PaperType','a4letter', ...
  	'Position',[0.1 0.1 0.8 0.8], ...
	'Tag','Fig1');

% --------------- QUIT button -------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
   'ListboxTop',0, ...
   'ForegroundColor',[1 1 1], ...
   'BackgroundColor',[0 0 1], ...
   'TooltipString','Quit Fourier-Mellin transform', ...
   'Position',[0.88 0.02 0.09 0.09],...
   'Style','push', ...
   'String','Quit', ...
	'Tag','Exit','Callback','close(gcf)');

% --------------- HELP button ------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'Callback','fm_guifun help', ...
	'ListboxTop',0, ...
	'Position',[0.88 0.12 0.09 0.09], ...
    'String','Help', ...
    'TooltipString','Invoke help window ', ...
   'Tag','Help');

% ------------------- axes for image 1 ------------------------------------
h1 = axes('Parent',h0, ...
	'Box','on', ...
	'CameraUpVector',[0 1 0], ...
	'CameraUpVectorMode','manual', ...
	'Color',[1 1 1], ...
   'FontSize',8, ...
	'Position',[0.02726146220570011 0.5022156573116691 0.3779429987608426 0.4387001477104873], ...
   'Tag','Axes1', ...
   'NextPlot','add', ...
  	'XColor',[0 0 0], ...
	'XTickMode','manual', ...
	'YColor',[0 0 0], ...
	'YTickMode','manual', ...
   'ZColor',[0 0 0]);

h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Position',[0.4966216216216217 -0.03041825095057038 9.160254037844386], ...
	'Tag','Axes1Text4', ...
	'VerticalAlignment','cap');
set(get(h2,'Parent'),'XLabel',h2);

h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Position',[-0.02364864864864866 0.4942965779467681 9.160254037844386], ...
	'Rotation',90, ...
	'Tag','Axes1Text3', ...
	'VerticalAlignment','baseline');
set(get(h2,'Parent'),'YLabel',h2);

h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','right', ...
	'Position',[-0.07432432432432433 1.129277566539924 9.160254037844386], ...
	'Tag','Axes1Text2', ...
	'Visible','off');
set(get(h2,'Parent'),'ZLabel',h2);

h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Position',[0.4966216216216217 1.022813688212928 9.160254037844386], ...
	'Tag','Axes1Text1', ...
	'VerticalAlignment','bottom');
set(get(h2,'Parent'),'Title',h2);

% ------------------- axes for image 2 ------------------------------------

h1 = axes('Parent',h0, ...
	'Box','on', ...
	'CameraUpVector',[0 1 0], ...
	'CameraUpVectorMode','manual', ...
	'Color',[1 1 1], ...
   'FontSize',8, ...
	'Position',[0.02853598014888337 0.01329394387001477 0.3784119106699752 0.4401772525849335], ...
   'Tag','Axes2', ...
   'NextPlot','add', ...
	'XColor',[0 0 0], ...
	'XTickMode','manual', ...
	'YColor',[0 0 0], ...
	'YTickMode','manual', ...
	'ZColor',[0 0 0]);

% xlabel
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Interruptible','off', ...
	'Position',[0.4966216216216217 -0.03041825095057016 9.160254037844386], ...
	'Tag','Axes2Text4', ...
	'VerticalAlignment','cap');
set(get(h2,'Parent'),'XLabel',h2);

% ylabel
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Interruptible','off', ...
	'Position',[-0.02364864864864864 0.4942965779467681 9.160254037844386], ...
	'Rotation',90, ...
	'Tag','Axes2Text3', ...
	'VerticalAlignment','baseline');
set(get(h2,'Parent'),'YLabel',h2);

% zlabel
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','right', ...
	'Interruptible','off', ...
	'Position',[-0.0777027027027027 2.247148288973384 9.160254037844386], ...
	'Tag','Axes2Text2', ...
	'Visible','off');
set(get(h2,'Parent'),'ZLabel',h2);

% title
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Interruptible','off', ...
	'Position',[0.4966216216216217 1.022813688212928 9.160254037844386], ...
	'Tag','Axes2Text1', ...
	'VerticalAlignment','bottom');
set(get(h2,'Parent'),'Title',h2);

%*****************************************

% display image names
% - IMAGE 1 ------------------------------------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[0.0060    0.9435    0.2245    0.0266], ...
	'String','Input 1', ...
	'Style','text', ...
	'Tag','Ref_im');



h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[0.1860    0.9435    0.2245    0.0266], ...
	'String','', ...
	'Style','text', ...
	'Tag','Ref_im_c');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[0.0060    0.4535    0.2246    0.0281], ...
	'String','Input 2', ...
	'Style','text', ...
	'Tag','Sens_im');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[0.1860    0.4535    0.2246    0.0281], ...
	'String','', ...
	'Style','text', ...
	'Tag','Sens_im_c');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[0.83    0.68    0.15    0.0281], ...
	'String','Registered Image', ...
	'Style','text', ...
	'Tag','Reg_im');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[0.67    0.68    0.15    0.0281], ...
	'String','', ...
	'Style','text', ...
	'Tag','Reg_im_c');

%*****************************************
% button for loading image 1
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'Callback','fm_guifun loadA; fm_guifun RotateScaleCropInput2; fm_guifun input1Select; fm_guifun input2Select;', ...
	'ListboxTop',0, ...
	'Position',[0.419 0.83 0.1 0.1], ...
   'String','Load Input 1', ...
   'TooltipString','Load image 1', ...
   'Tag','Pushbutton2');

% pop-up menu for image 1 selection ------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun input1Select', ...
	'ListboxTop',0, ...
	'Position',[0.42 0.7735 0.1 0.04], ...
	'String','Input|Windowed Input|Magnitude Spectrum|Phase Spectrum|Log-Polar|Windowed Log-polar|Invariant Features', ...
    'Style','popupmenu', ...
    'TooltipString','Input 1 analysis selection', ...
	'Tag','input1analysis', ...
	'Value',1);

% button for loading image 2
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'Callback','fm_guifun loadB; fm_guifun RotateScaleCropInput2; fm_guifun input1Select; fm_guifun input2Select;', ...
	'ListboxTop',0, ...
	'Position',[0.419 0.34 0.1 0.1], ...
   'String','Load Input 2', ...
   'TooltipString','Load image 2', ...
   'Tag','Pushbutton2');

% pop-up menu for image 2 selection ------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun input2Select', ...
	'ListboxTop',0, ...
	'Position',[0.42 0.2835 0.1 0.04], ...
	'String','Input|Windowed Input|Magnitude Spectrum|Phase Spectrum|Log-Polar|Windowed Log-polar|Invariant Features', ...
    'Style','popupmenu', ...
    'TooltipString','Input 2 analysis selection', ...
	'Tag','input2analysis', ...
	'Value',1);

% ----------- Rotate input 2 edit box ----------------------------------
  h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'HorizontalAlignment','left', ...
	'Position',[0.42 0.23 0.05 0.04], ...
   'String','Rotate image 2', ...
   'TooltipString','Apply a rotation to image 2', ...
	'Style','text', ...
	'Tag','RotText');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',[1 1 1], ...
    'Callback','fm_guifun RotateScaleCropInput2; fm_guifun input2Select; fm_guifun input1Select;',...
    'Position',[0.47 0.24 0.05 0.03], ...
	'String','0', ...
	'Style','edit', ...
	'Tag','RotInput2');

% ----------- Scale input 2 edit box ----------------------------------
  h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'HorizontalAlignment','left', ...
	'Position',[0.42 0.18 0.05 0.04], ...
   'String','Scale image 2', ...
   'TooltipString','Apply a scale change to image 2', ...
	'Style','text', ...
	'Tag','SclText');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',[1 1 1], ...
    'Callback','fm_guifun RotateScaleCropInput2; fm_guifun input2Select; fm_guifun input1Select;',...
    'Position',[0.47 0.19 0.05 0.03], ...
	'String','0', ...
	'Style','edit', ...
	'Tag','SclInput2');

% ----------- crop input 2 edit box ----------------------------------
%  h1 = uicontrol('Parent',h0, ...
%	'Units','normalized', ...
%	'BackgroundColor',colo, ...
%	'HorizontalAlignment','left', ...
%	'Position',[0.42 0.13 0.05 0.04], ...
%   'String','Resize image 2', ...
%   'TooltipString','Size', ...
%	'Style','text', ...
%	'Tag','SclText');

%h1 = uicontrol('Parent',h0, ...
%	'Units','normalized', ...
%	'BackgroundColor',[1 1 1], ...
%    'Callback','fm_guifun RotateScaleCropInput2; fm_guifun input2Select; fm_guifun input1Select;',...
%    'Position',[0.47 0.14 0.05 0.03], ...
%	'String','100', ...
%	'Style','edit', ...
%	'Tag','CropInput2');

% ----------------- AUTOCROP check box ------------------------
h1 = uicontrol('Parent',h0, ...
   'Units','normalized', ...
  	'BackgroundColor',colo, ...
	'Callback','fm_guifun autocrop; fm_guifun RotateScaleCropInput2; fm_guifun input2Select; fm_guifun input1Select;', ...
	'ListboxTop',0, ...
	'Position',[0.42 0.13 0.07 0.05], ...
   'String','Autocrop', ...
   'TooltipString','Enable auto-cropping of inputs', ...
   'Style','checkbox',...
	'Tag','cb_autocrop');


% ----------------------- REGISTER button ------------------------------
% button for registering images
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'Callback','fm_guifun register; fm_guifun regSelect;', ...
	'ListboxTop',0, ...
	'Position',[0.79 0.76 0.16 0.12], ...
   'String','REGISTER', ...
   'TooltipString','Register', ...
   'Tag','Pushbutton1');

% slider to control how many scale/rotation peaks to look at in finding the solution
h1 = uicontrol('Parent',h0, ...
    'Units','normalized', ...
	'BackgroundColor',colo, ...
    'Callback','fm_guifun setPerformanceLevel', ...
	'FontSize',10, ...
	'ForegroundColor',[1 0 0], ...    
    'Min',1,'Max',200,'Value',1, ...
    'Position',[0.79 0.74 0.16 0.02], ...
    'Style','slider',...
    'Tag','performLevel', ...
    'TooltipString','Number of phase correlation peaks to search');


% ------------------- axes for REGISTERED image ------------------------------------

h1 = axes('Parent',h0, ...
	'Box','on', ...
	'CameraUpVector',[0 1 0], ...
	'CameraUpVectorMode','manual', ...
	'Color',[1 1 1], ...
   'FontSize',8, ...
	'Position',[0.575 0.225 0.4 0.45], ...
   'Tag','Axes3', ...
   'NextPlot','add', ...
	'XColor',[0 0 0], ...
	'XTickMode','manual', ...
	'YColor',[0 0 0], ...
	'YTickMode','manual', ...
	'ZColor',[0 0 0]);

% xlabel
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Interruptible','off', ...
	'Position',[0.4966216216216217 -0.03041825095057016 9.160254037844386], ...
	'Tag','Axes3Text4', ...
	'VerticalAlignment','cap');
set(get(h2,'Parent'),'XLabel',h2);

% ylabel
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Interruptible','off', ...
	'Position',[-0.02364864864864864 0.4942965779467681 9.160254037844386], ...
	'Rotation',90, ...
	'Tag','Axes3Text3', ...
	'VerticalAlignment','baseline');
set(get(h2,'Parent'),'YLabel',h2);

% zlabel
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','right', ...
	'Interruptible','off', ...
	'Position',[-0.0777027027027027 2.247148288973384 9.160254037844386], ...
	'Tag','Axes3Text2', ...
	'Visible','off');
set(get(h2,'Parent'),'ZLabel',h2);

% title
h2 = text('Parent',h1, ...
	'ButtonDownFcn','ctlpanel SelectMoveResize', ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Interruptible','off', ...
	'Position',[0.4966216216216217 1.022813688212928 9.160254037844386], ...
	'Tag','Axes3Text1', ...
	'VerticalAlignment','bottom');
set(get(h2,'Parent'),'Title',h2);

% -------------- POP-UP MENUS......... -------------------------------------

% static text for window type
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.58 0.86 0.07 0.04], ...
	'String','Window :', ...
	'Style','text', ...
	'Tag','textWindowType', ...
    'TooltipString','Window Type');

% ------------- Pop-up menu for window selection -------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun SelectWindow; fm_guifun RotateScaleCropInput2; fm_guifun input1Select; fm_guifun input2Select;', ...
	'ListboxTop',0, ...
	'Position',[0.66 0.86 0.1 0.04], ...
	'String','none|bartlett|blackman|boxcar|chebwin|hamming|hann|kaiser|triang', ...
    'Style','popupmenu', ...
    'TooltipString','Choice of window for FFT calculation', ...
	'Tag','FFTwindow', ...
	'Value',1);

% static text for rotation interpolation type
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.58 0.82 0.07 0.04], ...
	'String','Rotation :', ...
	'Style','text', ...
	'Tag','textRotInterp', ...
    'TooltipString','Rotation Interpolation');

% ------------- Pop-up menu for rotation interpolation type -------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun SetRotInterp; fm_guifun RotateScaleCropInput2; fm_guifun input1Select; fm_guifun input2Select;', ...
	'ListboxTop',0, ...
	'Position',[0.66 0.82 0.1 0.04], ...
	'String','nearest|bilinear|bicubic', ...
    'Style','popupmenu', ...
    'TooltipString','Rotation interpolation', ...
	'Tag','RotInterp', ...
	'Value',1);

% static text for scale interpolation type
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.58 0.78 0.07 0.04], ...
	'String','Scale :', ...
	'Style','text', ...
	'Tag','textSclInterp', ...
    'TooltipString','Scale Interpolation');

% ------------- Pop-up menu for scale interpolation type -------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun SetSclInterp; fm_guifun RotateScaleCropInput2; fm_guifun input1Select; fm_guifun input2Select;', ...
	'ListboxTop',0, ...
	'Position',[0.66 0.78 0.1 0.04], ...
	'String','nearest|bilinear|bicubic', ...
    'Style','popupmenu', ...
    'TooltipString','Scale interpolation', ...
	'Tag','SclInterp', ...
	'Value',1);

% static text for log-polar interpolation type
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.58 0.74 0.07 0.04], ...
	'String','Log-polar :', ...
	'Style','text', ...
	'Tag','textLPInterp', ...
    'TooltipString','Log-Polar Interpolation');

% ------------- Pop-up menu for logpolar conversion selection -------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun SetLogPolInterp; fm_guifun input1Select; fm_guifun input2Select;', ...
	'ListboxTop',0, ...
	'Position',[0.66 0.74 0.1 0.04], ...
	'String','nearest|bilinear|bicubic', ...
    'Style','popupmenu', ...
    'TooltipString','Log-polar interpolation', ...
	'Tag','LogPolInterp', ...
	'Value',1);

% ------------- Check-box for Text Display
h1 = uicontrol('Parent',h0, ...
   'Units','normalized', ...
  	'BackgroundColor',colo, ...
	'Callback','fm_guifun dispText', ...
	'ListboxTop',0, ...
	'Position',[0.8 0.88 0.05 0.05], ...
   'String','Debug', ...
   'TooltipString','Display text information', ...
   'Style','checkbox',...
	'Tag','cb_dispText');


% pop-up menu for registered image selection -------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'Callback','fm_guifun regSelect', ...
	'ListboxTop',0, ...
	'Position',[0.6 0.675 0.15 0.04], ...
	'String','Registered|Registered Image 1|Registered Image 2|Log-Polar Phase Correlation|Spatial Phase Correlation', ...
    'Style','popupmenu', ...
    'TooltipString','Registered image analysis selection', ...
	'Tag','reganalysis', ...
	'Value',1);

% static text boxes for displaying registration information ----------------------
% -------------------- TRANSLATION output -------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.6 0.15 0.07 0.02], ...
	'String','Translation', ...
	'Style','text', ...
	'Tag','textTransOut', ...
    'TooltipString','');

h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.68 0.15 0.07 0.02], ...
	'String','', ...
	'Style','text', ...
	'Tag','TransOut', ...
    'TooltipString','Registered Translation');

% -------------------- ROTATION output -------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.6 0.10 0.07 0.02], ...
	'String','Rotation', ...
	'Style','text', ...
	'Tag','textRotOut', ...
    'TooltipString','');

  h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.68 0.1 0.07 0.02], ...
	'String','', ...
	'Style','text', ...
	'Tag','RotOut', ...
    'TooltipString','Registered Rotation');

% -------------------- SCALE output -------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.6 0.05 0.07 0.02], ...
	'String','Scale', ...
	'Style','text', ...
	'Tag','textScaleOut', ...
    'TooltipString','');

  h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.68 0.05 0.07 0.02], ...
	'String','', ...
	'Style','text', ...
	'Tag','ScaleOut', ...
    'TooltipString','Registered Scale');

% ------------------------ DISPLAY PEAK CORRELATIONS ---------------------
% -------------------- SCALE output -------------------------------------
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.75 0.15 0.07 0.02], ...
	'String','', ...
	'Style','text', ...
	'Tag','TransPeakOut', ...
    'TooltipString','Translation Peak Phase Correlation');

  h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',12, ...
   'ListboxTop',0, ...
  	'ForegroundColor',[0 0 1], ...
   'HorizontalAlignment','left', ...
   'Position',[0.75 0.06 0.07 0.04], ...
	'String','', ...
	'Style','text', ...
	'Tag','RSPeakOut', ...
    'TooltipString','Rotation/Scale Peak Phase Correlation');

% message saying what this is and who I am...
h1 = uicontrol('Parent',h0, ...
	'Units','normalized', ...
	'BackgroundColor',colo, ...
	'FontSize',10, ...
	'ForegroundColor',[1 0 0], ...
	'ListboxTop',0, ...
	'Position',[0.53  0.92 0.43 0.04], ...
   'Style','text', ...
   'String','Fourier-Mellin Transform GUI ... Adam Wilmer (aiw99r@ecs.soton.ac.uk)', ...
	'Tag','Text');
