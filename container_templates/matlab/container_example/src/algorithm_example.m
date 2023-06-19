% algorithm_example.md - template to read/write to local directory 
% or S3 bucket
% ---------------------------------------------------------------------
%     
% Example:
%  
%  in terminal -
%  matlab -nodisplay -r "cd src; algorithm_example <manifest_in path>"
%  
%  in MATLAB console - 
%  setenv("PROCESSING_DROPBOX","<dropbox path>");
%  algorithm_example(<manifest_in path>)
% 
% Notes:
%   - If running from the terminal add
%   export PROCESSING_DROPBOX="<dropbox path>" to your bash profile
%   - <path> may either be the path to the local directory e.g.
%   '../container_example_data/dropbox/input_manifest_20220923t000000_local.json' 
%   or the path to the s3 bucket (e.g. s3://bucketname/).
%   - For instructions on using this code with Docker refer to
%   docker.md
%
% Authors: Laura Sandoval, Matt Watwood, Gavin Medley
%
% External libraries used:
%
% Stefan Stoll (2022). MD5 signature of a file 
% (https://www.mathworks.com/matlabcentral/fileexchange/5498-md5-signature-of-a-file), 
% MATLAB Central File Exchange. Retrieved November 9, 2022.
% 
% Matthew Spaethe (2022). Matlab logging facility 
% (https://www.mathworks.com/matlabcentral/fileexchange/42078-matlab-logging-facility), 
% MATLAB Central File Exchange. Retrieved November 15, 2022.
% ---------------------------------------------------------------------

function []=algorithm_example(manifest_in, varargin)
% Requires 1 input and has optional input.

addpath('logging')
import logging.*

% get the global LogManager
LogManager.getLogManager();

% add a logger instance to this script
logger = Logger.getLogger('AlgorithmExample');
logger.setLevel( Level.ALL );

defaultInt = 2;

logger.info(strcat('Command executed in: ', mfilename('filename'),'.m'));
logger.info('Parsing CLI arguments.');

p = inputParser;
addRequired(p,'manifest_in',@ischar);
addOptional(p,'multiplier',defaultInt,@isnumeric);
parse(p,manifest_in,varargin{:});

logger.info(strcat('Manifest file to read: ', p.Results.manifest_in));
logger.info(strcat('Additional options passed are multiplier = ', ...
    num2str(p.Results.multiplier)));

% only want 1 optional input at most
numvarargs = length(varargin);
if numvarargs > 1
    error('my_script:TooManyInputs', ...
        'requires at most 1 optional inputs');
end

processing_dropbox = getenv('PROCESSING_DROPBOX');

str = fileread(p.Results.manifest_in);
json_content = jsondecode(str);

logger.info(strcat('Manifest type is ', json_content.manifest_type));
logger.info(strcat('Manifest contains ', json_content.files.filename));

% Get the data from each file and put data into some format that will be 
% used in the algorithm
for n = 1 : length(json_content.files)

    checksum = json_content.files(n).checksum;
    filename = json_content.files(n).filename;

    assert(isequal(checksum,md5(filename)), ...
        'algorithm_example:checksumError','Checksums do not match!')
    logger.info(strcat('Checksum matches for ', filename));

  data_in = h5read(filename,'/HDFEOS/SWATHS/Swath1/DataField/Temperature');
  logger.info(strcat('Found input data in HDF5 file:',mat2str(data_in)))

end

%fake data product to write as output
data_out = data_in * p.Results.multiplier;

%Create hdf5 data
output_filepath = append(processing_dropbox, 'example_output.h5');
h5create(output_filepath,'/data/array1',[height(data_out) width(data_out)]);

logger.info(strcat('Writing output file: ', output_filepath))

h5write(output_filepath, '/data/array1', data_out);
h5writeatt(output_filepath,'/','someattr','hello, world');

% Need this to create empty group; if you are creating a group 
% that contains data, then use h5create and h5write
plist = 'H5P_DEFAULT';
fid = H5F.open(output_filepath,'H5F_ACC_RDWR',plist);
gid = H5G.create(fid,'new_group',plist,plist,plist);
H5G.close(gid);
H5F.close(fid);

json_out = append(processing_dropbox, 'output_manifest_20220923t111111.json');

logger.info(strcat('Writing output manifest: ', json_out))

%Get checksum of written file
checksum_written = md5(output_filepath);

s1.filename = output_filepath;
s1.checksum = checksum_written;
struct_1 = jsonencode(s1);

s.manifest_type = "OUTPUT";
s.files = [struct_1];
s.configuration = string(nan);

fid = fopen(json_out,'w');

fprintf(fid,'%s',strrep(jsonencode(s, "PrettyPrint", true),'\',''));
fclose(fid);

logger.info('Algorithm complete. Exiting.')

exit
end
