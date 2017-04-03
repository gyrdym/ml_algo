// Copyright (c) 2017, Ilya Gyrdymov. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import 'package:dart_ml/dart_ml.dart';
import 'package:dart_ml/src/vector_operations.dart' as vectors;

main() {
  LinearRegression predictor = new LinearRegression();

  List<List<double>> features = [
    [113242.0, 23221.32423234, 345322.7, 54566.78333, 34536.0, 345443.0]
  ];

  List<double> labels = [4.0];

  predictor.train(features, labels);

  print("weights: ${predictor.weights}");
  print("checking, original label: ${labels.first}, predicted label: ${vectors.scalarMult(features.first, predictor.weights)}");
}
