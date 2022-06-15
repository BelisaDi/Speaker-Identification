function isSpeech = isSpeech(x,windowLength,overlapLength,pwrThreshold)
% Retorna los datos de la señal que están por encima de un umbral dado.
[segments,~] = buffer(x,windowLength,overlapLength,'nodelay');
pwr = pow2db(bandpower(segments));
isSpeech = (pwr > pwrThreshold);
end