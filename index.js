/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {BostonHousingDataset, featureDescriptions} from './data';
import * as normalization from './normalization';
import * as ui from './ui';

// Some hyperparameters for model training.
// https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week4/week3quiz1.md
// INFO: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9

var NUM_EPOCHS = 200;//es el número de veces que el modelo está expuesto al conjunto de datos de entrenamiento,debe ser una medidad a  discresion del volumen de data a evaluar
var BATCH_SIZE = 40;// es el número de instancias de entrenamiento mostradas al modelo antes de que se realice una actualización de peso.
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};

// Convert loaded data into tensors and creates normalized versions of the
// features.
export const arraysToTensors = () => {
    tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
    tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
    tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
    tensors.testTarget = tf.tensor2d(bostonData.testTarget);
    // Normalize mean and standard deviation of data.
    let {dataMean, dataStd} =
            normalization.determineMeanAndStddev(tensors.rawTrainFeatures);

    tensors.trainFeatures = normalization.normalizeTensor(
            tensors.rawTrainFeatures, dataMean, dataStd);
    tensors.testFeatures =
            normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
};

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export function linearRegressionModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [bostonData.numFeatures], units: 1}));

    model.summary();
    return model;
}
;

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export function multiLayerPerceptronRegressionModel1Hidden() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [bostonData.numFeatures],
        units: 50,
        activation: 'sigmoid',
        kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dense({units: 1}));

    model.summary();
    return model;
}
;

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
export function multiLayerPerceptronRegressionModel2Hidden() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [bostonData.numFeatures],
        units: 50,
        activation: 'sigmoid',
        kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dense(
            {units: 50, activation: 'sigmoid', kernelInitializer: 'leCunNormal'}));
    model.add(tf.layers.dense({units: 1}));

    model.summary();
    return model;
}
;


/**
 * Describe the current linear weights for a human to read.
 *
 * @param {Array} kernel Array of floats of length 12.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value feature weight.
 */
export function describeKerenelElements(kernel) {
    tf.util.assert(kernel.length == 12, `kernel must be a array of length 12, got ${kernel.length}`);
    const outList = [];
    for (let idx = 0; idx < kernel.length; idx++) {
        outList.push({description: featureDescriptions[idx], value: kernel[idx]});
    }
    return outList;
}

/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 * @param {boolean} weightsIllustration Whether to print info about the learned
 *  weights.
 */
export const run = async (model, weightsIllustration) => {
    await ui.updateStatus('Compilando modelo...');
    /*  https://keras.io/optimizers/ */
    model.compile({optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError'});
//https://stats.stackexchange.com/questions/187335/validation-error-less-than-training-error/187404#187404
    let trainLoss;
    let valLoss;
    await ui.updateStatus('Procesando entrenamiento...');
    //recive parameters
    NUM_EPOCHS = (document.getElementById('trainIterations')).value;
    console.log('trainIterations: ' + (document.getElementById('trainIterations')).value);

    BATCH_SIZE = parseInt((document.getElementById('batchSize')).value);
    console.log('batchSize: ' + BATCH_SIZE);
    
        await bostonData.loadData();

    ui.updateStatus('Data Cargada, convirtiendo a tensores');
    arraysToTensors();
    ui.updateStatus(
            'Datos estan disponibles como tensores.\n' +
            'Click para empezar a entrenar.');
    ui.updateBaselineStatus('La estimación de la pérdida de la línea de base');
    computeBaseline();
    await ui.setup();
    
    
    // 
    await model.fit(tensors.trainFeatures, tensors.trainTarget, {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                await ui.updateStatus(`Epoch ${epoch + 1} de  ${NUM_EPOCHS} completado.`);
                trainLoss = logs.loss;
                valLoss = logs.val_loss;
                await ui.plotData(epoch, trainLoss, valLoss);
                if (weightsIllustration) {
                    model.layers[0].getWeights()[0].data().then(kernelAsArr => {
                        const weightsList = describeKerenelElements(kernelAsArr);
                        ui.updateWeightDescription(weightsList);
                    });
                }
            }
        }
    });

    await ui.updateStatus('Procesando datos de prueba...');
    const result = model.evaluate(tensors.testFeatures, tensors.testTarget, {batchSize: BATCH_SIZE});
    const testLoss = result.dataSync()[0];
    await ui.updateStatus(
            `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
            `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
            `Test-set loss: ${testLoss.toFixed(4)}`);
};

export const computeBaseline = () => {
    const avgPrice = tf.mean(tensors.trainTarget);
    console.log(`Promedio : ${avgPrice.dataSync()}`);
    const baseline = tf.mean(tf.pow(tf.sub(tensors.testTarget, avgPrice), 2));
    console.log(`Desviación de la media al cuadrado a la linea de Base: ${baseline.dataSync()}`);
    const baselineMsg = `Desviación de la media al cuadrado a la linea de Base (meanSquaredError) es ${
            baseline.dataSync()[0].toFixed(2)}`;
    ui.updateBaselineStatus(baselineMsg);
};

document.addEventListener('DOMContentLoaded', async () => {
    await bostonData.loadData();
    ui.updateStatus('Data Cargada, ... convirtiendo a tensores');
    arraysToTensors();
    ui.updateStatus(
            'Datos estan disponibles como tensores.\n' +
            'Click para empezar a entrenar.');
    ui.updateBaselineStatus('La estimación de la pérdida de la línea de base');
    computeBaseline();
    await ui.setup();
}, false);
