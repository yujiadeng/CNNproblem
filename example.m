function res = example(varargin)
    run vlfeat/toolbox/vl_setup ;
    run matconvnet/matlab/vl_setupnn ;
    addpath matconvnet/examples ;
    rng(1)
    % SGD settings
    trainOpts.batchSize =50;
    trainOpts.numEpochs = 20 ;
    trainOpts.continue = true ;
    trainOpts.gpus = [] ;
    trainOpts.learningRate = 0.001 ;
    
    load example.mat % the data has been normalized
    imageMean = mean(imdb.images.data,3); 
    for j= 1:size(imdb.images.data,3)
        imdb.images.data(:,:,j) = imdb.images.data(:,:,j) - imageMean ;
    end
    % Training
    net = initialize();
    trainOpts = vl_argparse(trainOpts, varargin);
    trainOpts.expDir = 'data/example';
    [net, info] = cnn_train(net, imdb, @getBatch, trainOpts);
    net.layers(end) = [];
    net.imageMean = imageMean;
    save('data/example/example.mat', '-struct', 'net');
    % Result analyze
    [Accu, Sens, Spec] = Apply(net, imdb.test);
    res = [Accu, Sens, Spec];     
end

function net = initialize()
    f=1/100 ;
    net.layers = {} ;
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(3,3,1,16, 'single'), zeros(1, 16, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(3,3,16,32, 'single'), zeros(1,32,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 1) ;

    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(7,7,32,2, 'single'), zeros(1,2,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;
    net.layers{end+1} = struct('type', 'softmaxloss') ;
    net = vl_simplenn_tidy(net) ;
end

function [im, labels] = getBatch(imdb, batch)
    im = 256 * reshape(imdb.images.data, 32 , 32, 1, []);
    im = im(:,:,:,batch);
    labels = imdb.images.label(1,batch);
end
function pred = predict(net,test)
    im = test.data;
    for i = 1:size(test.data,3)
        im(:,:,i) = (test.data(:,:,i) - net.imageMean);
    end
    im = 256 * reshape(im, 32, 32, 1, []);
    score = zeros(size(im,4));
    pred = zeros(size(im,4));
    for i = 1:size(im, 4)
        res = vl_simplenn(net, im(:,:,:,i));
        [score(i),pred(i)]=max(res(end).x(1,1,:));
    end
end
function [Accu, Sens, Spec] = evaluate(pred, test)
    TP=sum(pred(test.label==2)==2);
    FN=sum(pred(test.label==2)==1);
    TN=sum(pred(test.label==1)==1);
    FP=sum(pred(test.label==1)==2);
    Sens=TP/(TP+FN);
    Spec=TN/(TN+FP);
    Accu=(TP+TN)/(TP+TN+FP+FN);
    fprintf('Accuracy: %.3f, Sensitivity: %.3f, Specificity: %.3f\n',Accu, Sens, Spec);       
end
function [Accu, Sens, Spec] = Apply(net,test)
    pred = predict(net,test);
    [Accu, Sens, Spec] = evaluate(pred, test);
end