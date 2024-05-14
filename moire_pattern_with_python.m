% I = imread('CFA_bayer.tif');
% J = demosaic(I,'gbrg');
% imshow(I);

%% 调用Zemax实现图像模拟，将PNG结果保存在当前文件夹下
OpenFile_and_ImageSimulation([]);



function [ r ] = OpenFile_and_ImageSimulation( args )

if ~exist('args', 'var')
    args = [];
end

% Initialize the OpticStudio connection
TheApplication = InitConnection();
if isempty(TheApplication)
    % failed to initialize a connection
    r = [];
else
    try
        r = BeginApplication(TheApplication, args);
        CleanupConnection(TheApplication);        
    catch err
        CleanupConnection(TheApplication);
        rethrow(err);
    end
end
end


function [ r ] = BeginApplication(TheApplication, args)

    import ZOSAPI.*;

    % creates a new API directory
    apiPath = System.String.Concat(TheApplication.SamplesDir, '\API\Matlab');
    if (exist(char(apiPath)) == 0) mkdir(char(apiPath)); end;
    
    % 载入取像系统zmx文档
    TheSystem = TheApplication.CreateNewSystem(ZOSAPI.SystemType.Sequential);
    TheSystem.LoadFile(System.String.Concat('C:\Users\zilin\Documents\Zemax\Samples\Calculate PSF_Double Gauss 28 degree field.zos'), false);
            
    % 图像模拟
    image_analysis = TheSystem.Analyses.New_ImageSimulation();
    image_settings = image_analysis.GetSettings();
    image_settings.InputFile = 'C:\Users\zilin\PycharmProjects\Moire_Attack-main\LCD子像素旋转变换后图片.PNG';
    image_settings.FieldHeight = 40;
    image_settings.SetWavelength123();

    image_settings.PupilSampling = ZOSAPI.Analysis.SampleSizes.S_32x32;
    image_settings.ImageSampling = ZOSAPI.Analysis.SampleSizes.S_32x32;
    image_settings.PSFXPoints = 5;
    image_settings.PSFYPoints = 5;
    image_settings.UsePolarization = false;
    
    image_settings.Aberrations = ZOSAPI.Analysis.Settings.ExtendedScene.ISAberrationTypes.Geometric;
    image_settings.ApplyFixedApertures = false;
    image_settings.UseRelativeIllumination = false;

    image_settings.ShowAs = ZOSAPI.Analysis.Settings.ExtendedScene.ISShowAsTypes.SimulatedImage;
    image_settings.OutputFile = 'C:\Users\zilin\PycharmProjects\Moire_Attack-main\LCD子像素旋转_psf卷积后图像_zemax.PNG';

    image_analysis.ApplyAndWaitForCompletion();
    
    
    r = [];
end



function app = InitConnection()

import System.Reflection.*;

% Find the installed version of OpticStudio.
zemaxData = winqueryreg('HKEY_CURRENT_USER', 'Software\Zemax', 'ZemaxRoot');
NetHelper = strcat(zemaxData, '\ZOS-API\Libraries\ZOSAPI_NetHelper.dll');
% Note -- uncomment the following line to use a custom NetHelper path
% NetHelper = 'C:\Users\Documents\Zemax\ZOS-API\Libraries\ZOSAPI_NetHelper.dll';
NET.addAssembly(NetHelper);

success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize();
% Note -- uncomment the following line to use a custom initialization path
% success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize('C:\Program Files\OpticStudio\');
if success == 1
    LogMessage(strcat('Found OpticStudio at: ', char(ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory())));
else
    app = [];
    return;
end

% Now load the ZOS-API assemblies
NET.addAssembly(AssemblyName('ZOSAPI_Interfaces'));
NET.addAssembly(AssemblyName('ZOSAPI'));

% Create the initial connection class
TheConnection = ZOSAPI.ZOSAPI_Connection();

% Attempt to create a Standalone connection

% NOTE - if this fails with a message like 'Unable to load one or more of
% the requested types', it is usually caused by try to connect to a 32-bit
% version of OpticStudio from a 64-bit version of MATLAB (or vice-versa).
% This is an issue with how MATLAB interfaces with .NET, and the only
% current workaround is to use 32- or 64-bit versions of both applications.
app = TheConnection.CreateNewApplication();
if isempty(app)
   HandleError('An unknown connection error occurred!');
end
if ~app.IsValidLicenseForAPI
    HandleError('License check failed!');
    app = [];
end

end

function LogMessage(msg)
disp(msg);
end

function HandleError(error)
ME = MXException(error);
throw(ME);
end

function  CleanupConnection(TheApplication)
% Note - this will close down the connection.

% If you want to keep the application open, you should skip this step
% and store the instance somewhere instead.
TheApplication.CloseApplication();
end
