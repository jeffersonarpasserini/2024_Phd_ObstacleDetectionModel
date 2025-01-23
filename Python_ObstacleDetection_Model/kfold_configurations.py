# ------- 1a configuração  testada -------------
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))  # Define explicitamente a entrada
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Saída binária

        # Compilar o modelo
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        # Treinar o modelo com as features extraídas
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Adicionar o callback para matriz de confusão
        confusion_matrix_callback = ConfusionMatrixCallback(X_validation, y_validation, RESULT_PATH_TRAINING, split_index)

        model.fit(
            X_train,
            y_train,
            epochs=35,
            batch_size=64,
            validation_data=(X_validation, y_validation),
            callbacks=[early_stopping, confusion_matrix_callback]
        )

