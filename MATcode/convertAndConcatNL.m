function neuralDataFullSession = convertAndConcatNL(startFile, stopFile)
% The script takes all the files from that day and concatenates them into a bin file.
% inputs: 
%   startFile - Name of the fisrt file (i.e. 'NEUR0000.DT2')
%   endFile - Name of the end file
% outputs:
%   fullSessionFile
fSample = 32e3;

fileToConvert = startFile;
fullSessionFile = [];
rawNumFile = str2double(startFile(5:8));

while ~strcmp(fileToConvert(1:end-4),stopFile)
    currFile = convertNeurologger(fileToConvert);
    fullSessionFile = [fullSessionFile currFile];
    rawNumFile = rawNumFile + 1;
    fileToConvert = ['NEUR', sprintf('%04d', rawNumFile), '.DT2'];
end

    lastFile = convertNeurologger([stopFile, ext]);
    fullSessionFile = [fullSessionFile lastFile];

    %filters
    lowpass = 300;
    highpass = 6000;
    neuralDataFullSession = bandpass(fullSessionFile, [lowpass highpass], fSample); % band pass filter
end