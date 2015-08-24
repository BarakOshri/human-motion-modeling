function many_xyz = many_exp2xyz(channels)
    skel = buildskel_mit();
    many_xyz = exp2xyz(skel, channels(1,:))
    for i = 2:size(channels, 1)
        many_xyz = vertcat(many_xyz, exp2xyz(skel, channels(i,:)));
    end
end