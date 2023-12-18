clc; clear; close all;

%% load exel with NL data
global NLpath
global recfilePath
NLfollow = readtable('C:\Zaidel\Rats\NL\NL_Follow_2023.xlsx', ...
    'Sheet', 'Sheet1', 'ReadVariableNames', true);
NLpath = 'C:\Zaidel\Rats\NL';
recfilePath = fullfile(NLpath, 'Rec Data');
% recfilePath = 'C:\Zaidel\Rats\Data\Recordings22';

%%
for i = 1:1%height(NLrecData)
    fullSessionFile = convertAndConcatNL(NLfollow.StartFile{i}, NLfollow.EndFile{i});
    
    ratName = [num2str(NLfollow.Rat_(i)), ' - ', NLfollow.RatName{i}];
    folderPath = fullfile(NLpath, 'Data', ratName, NLfollow.Date{i});
    fileName = ['NLrec', NLfollow.Date{i}, NLfollow.RatName{i}, '.bin'];
    filePath = fullfile(folderPath, filePath);

    fid = fopenf(filePath, 'w');
    fwrite(fid, fullSessionFile,'int16');
    fclose(fid);
    save([filePath, '.bin'], 'fullSessionFile', '-v7.3')

end