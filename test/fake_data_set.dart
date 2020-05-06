import 'package:ml_dataframe/ml_dataframe.dart';

final fakeDataSet = DataFrame.fromSeries([
  Series('col_1', <int>[10, 90, 23, 55]),
  Series('col_2', <int>[20, 51, 40, 10]),
  Series('col_3', <int>[1, 0, 0, 1], isDiscrete: true),
  Series('col_4', <int>[0, 0, 1, 0], isDiscrete: true),
  Series('col_5', <int>[0, 1, 0, 0], isDiscrete: true),
  Series('col_6', <int>[30, 34, 90, 22]),
  Series('col_7', <int>[40, 31, 50, 80]),
  Series('col_8', <int>[0, 0, 1, 2], isDiscrete: true),
]);
