function model = TrainModel()
    % Función que retorna el modelo knn entrenado para la clasificación de
    % los audios.
    dataDir = "Audios"; % Carpeta de la base de datos creada.
    audioFiles = audioDatastore(dataDir, "IncludeSubfolders",true, ...
        "FileExtensions",".wav", "LabelSource","foldernames");
    
    rng(1234); % Semilla
    [audioTrain, ~] = splitEachLabel(audioFiles, 0.7, 'randomized'); % Separación datos de prueba y entrenamiento

    [~,audioInfo] = read(audioTrain); % Se leen los audios de entrenamiento.
    reset(audioTrain);

    Fs = audioInfo.SampleRate; % Frecuencia de los audios.
    windowLength = round(0.03*Fs); 
    overlapLength = round(0.015*Fs);

    % Se crean los vectores de características y de etiquetas.
    features = [];
    labels = [];

    while hasdata(audioTrain)
       [x,audioInfo] = read(audioTrain); 
       
       x = x./max(x); % Normalización de los datos

       % Se hallan los coeficientes ceptrales de la señal con la ventana de Hanning
       % y la frecuencia fundamental de la señal pata encontrar el
       % rasgo que identifica la señal.
       melCeptrum = mfcc(x,Fs,"Window",hanning(windowLength,"periodic"),"OverlapLength",overlapLength); 
       Pitch = pitch(x,Fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
       feat = [melCeptrum,Pitch];

       Speech = isSpeech(x,windowLength,overlapLength,-40); % Datos de la señal por encima del umbral dado.

       feat(~Speech,:) = []; % Elimina componentes de la señal que no están por enxima del umbral dado.
       label = repelem(audioInfo.Label,size(feat,1)); % Replica los elementos del label del audio dado.

       features = [features;feat]; % Se agrega la característica del audio a las características.
       labels = [labels,label]; % Se agrega la etiqueta correspondiente.

    end

    % Se estandarizan las características.
    M = mean(features,1);
    S = std(features,[],1);
    features = (features-M)./S;

    % Se crea el modelo adecuado para los audios dados.
    model = fitcknn(features,labels,"Distance","cityblock","NumNeighbors",1,...
     "DistanceWeight","equal","BreakTies", "smallest", "NSMethod", "exhaustive","Standardize",...
     false,"ClassNames",unique(labels));

    % Se hace validación cruzada para observar el correcto comportamiento
    % del modelo.
    k = 5;
    group = labels;
    c = cvpartition(group,'KFold',k); % 5-fold stratified cross validation
    partitionedModel = crossval(model,'CVPartition',c);  % Partición del modelo.
    validationAccuracy = 1 - kfoldLoss(partitionedModel,'LossFun','ClassifError'); % Accuracy de validación
     
%     fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
end