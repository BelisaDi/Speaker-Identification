function label = PredictionModel(model, x, Fs)
    % Retorna la etiqueta correspondiente a la señal de audio ingresada.
    % Input: modelo, señal de audio y frecuencia.
    
    % Se normaliza la señal y se establecen las dimensiones de las ventanas.
    x = x./max(x);
    windowLength = round(0.03*Fs);
    overlapLength = round(0.015*Fs);
    
   % Se hallan los coeficientes ceptrales de la señal con la ventana de Hanning
   % y la frecuencia fundamental de la señal pata encontrar el
   % rasgo que identifica la señal.
    melC = mfcc(x,Fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
    Pitch = pitch(x,Fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    feat = [melC,Pitch];

    Speech = isSpeech(x,windowLength,overlapLength,-40); % Datos de la señal por encima del umbral dado.

    feat(~Speech,:) = []; % Elimina componentes de la señal que no están por enxima del umbral dado.
    
    % Se estandariza la característica creada de las señal.   
    M = mean(feat,1);
    S = std(feat,[],1);
    feat = (feat-M)./S;
    
    % Se predice la categoría a la que corresponde el audio ingresado.
    prediction = predict(model,feat);
    prediction = categorical(string(prediction));
    label = mode(prediction);
end