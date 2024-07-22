/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class TestKMeans  {
    public static void main(String[] args) throws Exception {
        final Path dirPath = Paths.get("/knn/mnist");
        final String vectorFile = "train_vecs.vec";
        final String labelsFile = "train_labels.vec";
        final int numVectors = 60_000;
        final int numLabels = 10;
        final int numClusters = 10;

        try (MMapDirectory directory = new MMapDirectory(dirPath);
             IndexInput vectorInput = directory.openInput(vectorFile, IOContext.DEFAULT);
             IndexInput labelsInput = directory.openInput(labelsFile, IOContext.READONCE)) {

            final short[] trueVectorsLabels = new short[numVectors];
            for (int i = 0; i < numVectors; i++) {
                trueVectorsLabels[i] = (short) labelsInput.readInt();
            }

            RandomAccessVectorValues.Floats vectorValues =
                    new VectorsReader(vectorInput, numVectors, 784, 784 * Float.BYTES);
            KMeans.Results results = KMeans.cluster(vectorValues, VectorSimilarityFunction.EUCLIDEAN, numClusters);
            final short[] vectorsLabels = results.vectorCentroids();

            // Map clusters to labels
            int[][] labelCnts = new int[numClusters][numLabels];
            for (int i = 0; i < numLabels; i++) {
                labelCnts[i] = new int[numLabels];
            }
            for (int i = 0; i < numVectors; i++) {
                short cluster = vectorsLabels[i];
                short label = trueVectorsLabels[i];
                labelCnts[cluster][label]++;
            }
            short[] clusterLabels = new short[numClusters];
            for (int i = 0; i < numClusters; i++) {
                int maxCount = 0;
                for (short j = 0; j < numLabels; j++) {
                    if (labelCnts[i][j] > maxCount) {
                        maxCount = labelCnts[i][j];
                        clusterLabels[i] = j;
                    }
                }
                System.out.println(i + ": " + Arrays.toString(labelCnts[i])+ ", max label: " + clusterLabels[i]);
            }
            // calculate accuracy
            int[] transformedLabels = new int[trueVectorsLabels.length];
            for (int i = 0; i < transformedLabels.length; i++) {
                transformedLabels[i] = clusterLabels[vectorsLabels[i]];
            }
            int correctCount = 0;
            for (int i = 0; i < transformedLabels.length; i++) {
                if (transformedLabels[i] == trueVectorsLabels[i]) {
                    correctCount++;
                }
            }
            float accuracy = (float) correctCount / transformedLabels.length;
            System.out.println("Accuracy for " + numClusters + " clusters: " + accuracy);

        }

    }
}
