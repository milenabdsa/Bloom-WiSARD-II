Explicação da hierarquia de decisão

Primeira prioridade GHR e PC concordam
 `if ghr_pred == pc_pred: return ghr_pred`

Segunda prioridade GHR e LHR concordam
`elif ghr_pred == lhr_pred: return ghr_pred`

Terceira prioridade PC e LHR concordam
`elif pc_pred == lhr_pred: return pc_pred`

Quarta prioridade GHR e XOR concordam
`elif ghr_pred == xor_pred: return ghr_pred`

Quinta prioridade PC e XOR concordam
`elif pc_pred == xor_pred: return pc_pred`

Sexta prioridade LHR e XOR concordam
`elif lhr_pred == xor_pred: return lhr_pred`

Sétima prioridade GA e qualquer outra rede concordam
`elif ga_pred == [outra_rede]: return ga_pred`

Se nenhum consenso for encontrado, o sistema usa a rede com maior confiança, ou fallback de confiança.
E como um último recurso (fallback final), o sistema retorna a predição do discriminador padrão (PC). 

```python
confidence_scores = {
    'pc': pc_count_0 + pc_count_1,
    'lhr': lhr_count_0 + lhr_count_1,
    'ghr': ghr_count_0 + ghr_count_1,
    'ga': ga_count_0 + ga_count_1,
    'xor': xor_count_0 + xor_count_1
}
max_confidence_network = max(confidence_scores, key=confidence_scores.get)
``

Como usar?
```python
cd tournament

Você pode criar o modelo
```python
predictor = Model(parameters)

Fazer a predição
prediction = predictor.predict(pc)

E treinar o modelo
is_correct = predictor.predict_and_train(pc, actual_outcome)
```

Como testar o código?

(Testando com o rquivo I1.txt)
```bash
cd tournament
python main.py "../Dataset_pc_decimal/I1.txt" 4 4 4 4 4 5.0 2.0 2.0 2.0 3.0 4 4
```
(Testando com o rquivo S1.txt)
```bash
python main.py "../Dataset_pc_decimal/S1.txt" 4 4 4 4 4 5.0 2.0 2.0 2.0 3.0 4 4
```
(Testando com o rquivo M1.txt)
```bash
python main.py "../Dataset_pc_decimal/M1.txt" 4 4 4 4 4 5.0 2.0 2.0 2.0 3.0 4 4
```

Resultados:
`Results_accuracy/[arquivo]/[timestamp]-BTHOWeN-accuracy.png` é o gráfico de precisão
`Results_accuracy/[arquivo]/[timestamp]-BTHOWeN-accuracy.csv` são os dados de precisão
`true_bthowen_accuracy/[arquivo]-accuracy.csv`é o resultado final
