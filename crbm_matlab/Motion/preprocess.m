function Motion = preprocess(n1, activities_file)

skel = buildskel_mit();

Motion = {};
fid = fopen(activities_file);

count = 1;
tline = fgetl(fid);
while ischar(tline)
    activity = importdata(char(tline));
    tline = fgetl(fid);
    Motion{count} = activity;
    count = count + 1;

Motion = preprocess1(n1, Motion, skel);
end
