// Copyright (c) 2017, Ilya Gyrdymov. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import 'package:dart_ml/dart_ml.dart';

main() {
  LinearRegression predictor = new LinearRegression();

  List<List<double>> features = [
    [11.0, 2.324, 345.7],
    [43.0, 544.0, 0.0]
  ];

  List<double> labels = [4.0, 3.0];

  predictor.train(features, labels);

  print(predictor.weights);
}
