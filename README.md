# Machine learning with dart

Only linear regression with SGD is available now

## Usage

A simple usage example:
    
    import 'package:dart_ml/dart_ml.dart';
    
    SGDLinearRegressor predictor = new SGDLinearRegressor();
    
    List<List<double>> features = [
        [1.0, 2.0, 3.0, 223.3, 3.444, 23478.0],
        [4.0, 5.0, 7.2, 309.1, 237.98, 2345.0]
    ];
    
    List<double> labels = [4.0, 3.0];
    
    predictor.train(features, labels);
    
    print("weights: ${predictor.weights}");
    print("rmse (training) is: ${predictor.rmse}");