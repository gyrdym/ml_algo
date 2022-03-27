//import 'dart:convert';

import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

//import '../../helpers.dart';

void main() {
  group('KDTree', () {
    final data = [
      [3.43, 10.91, 11.62, -12.93, -11.66],
      [19.41, -4.96, 3.99, 16.35, 10.57],
      [11.30, 8.89, -17.66, -5.17, 16.20],
      [-8.13, -5.23, 18.01, 1.97, 9.08],
      [13.98, -8.21, 17.01, -5.14, 14.49],
      [-17.65, 13.10, 5.82, 8.61, 14.41],
      [4.16, -4.72, -3.71, -2.32, -13.70],
      [7.29, 11.16, -9.51, -1.89, -18.94],
      [19.81, 3.17, 14.27, 0.05, -17.93],
      [-9.63, 18.82, -14.40, -1.91, -6.58],
      [-10.95, -19.58, 9.05, 17.39, 3.30],
      [4.08, -13.19, -5.71, 18.56, -0.13],
      [2.79, -9.15, 6.56, -18.59, 13.53],
      [-7.56, 11.97, 6.55, -7.54, 15.90],
      [-15.97, -15.95, 7.71, 9.70, 16.94],
      [-15.01, 16.12, -10.42, -17.61, 6.27],
      [7.63, -10.70, 15.09, 10.25, -18.16],
      [0.05, 9.74, 7.08, 15.49, -17.99],
      [-6.48, 1.10, 9.28, 0.90, 6.09],
      [-9.88, -5.66, -16.15, 4.46, 2.34],
    ];

    final query = Vector.fromList([-9.88, -5.66, -16.15, 4.46, 2.34]);

    final distances = [
      Vector.fromList([3.43, 10.91, 11.62, -12.93, -11.66]).distanceTo(query),
      Vector.fromList([19.41, -4.96, 3.99, 16.35, 10.57]).distanceTo(query),
      Vector.fromList([11.30, 8.89, -17.66, -5.17, 16.20]).distanceTo(query),
      Vector.fromList([-8.13, -5.23, 18.01, 1.97, 9.08]).distanceTo(query),
      Vector.fromList([13.98, -8.21, 17.01, -5.14, 14.49]).distanceTo(query),
      Vector.fromList([-17.65, 13.10, 5.82, 8.61, 14.41]).distanceTo(query),
      Vector.fromList([4.16, -4.72, -3.71, -2.32, -13.70]).distanceTo(query),
      Vector.fromList([7.29, 11.16, -9.51, -1.89, -18.94]).distanceTo(query),
      Vector.fromList([19.81, 3.17, 14.27, 0.05, -17.93]).distanceTo(query),
      Vector.fromList([-9.63, 18.82, -14.40, -1.91, -6.58]).distanceTo(query),
      Vector.fromList([-10.95, -19.58, 9.05, 17.39, 3.30]).distanceTo(query),
      Vector.fromList([4.08, -13.19, -5.71, 18.56, -0.13]).distanceTo(query),
      Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]).distanceTo(query),
      Vector.fromList([-7.56, 11.97, 6.55, -7.54, 15.90]).distanceTo(query),
      Vector.fromList([-15.97, -15.95, 7.71, 9.70, 16.94]).distanceTo(query),
      Vector.fromList([-15.01, 16.12, -10.42, -17.61, 6.27]).distanceTo(query),
      Vector.fromList([7.63, -10.70, 15.09, 10.25, -18.16]).distanceTo(query),
      Vector.fromList([0.05, 9.74, 7.08, 15.49, -17.99]).distanceTo(query),
      Vector.fromList([-6.48, 1.10, 9.28, 0.90, 6.09]).distanceTo(query),
      Vector.fromList([-9.88, -5.66, -16.15, 4.46, 2.34]).distanceTo(query),
    ];

    // 0: 33.09060864402662
    // 1: 39.11458385008682,
    // 2: 34.23003613108167;
    // 3: 26.612617353374993;
    // 4: 20.423670786827675;
    // 5: 40.669621207668314;
    // 6: 33.662449711118605;
    // 7: 44.990910269030316;
    // 8: 42.87292938125955;
    // 9: 45.371062977550906;
    // 10: 41.26724998570784;
    // 11: 41.65641325613141;
    // 12: 0.0;
    // 13: 26.09399079267498;
    // 14: 34.80592470378389;
    // 15: 36.01945663144903;
    // 16: 43.98398182017352;
    // 17: 50.195289234794245,
    // 18: 25.171560833685394,
    // 19: 36.67385531240255,

    // [12, 4, 18, 13]

    final kdTree = KDTree(DataFrame(data, headerExists: false), leafSie: 3);

    test('should build a correct structure', () {
      print(distances..sort());
//      print(jsonEncode(kdTree.root.toJson()));
//      expect(kdTree.toJson(), {
//        'L': 3,
//        'R': {
//          'V': {
//            'DT': 'F32',
//            'D': [
//              -9.880000114440918,
//              -5.659999847412109,
//              -16.149999618530273,
//              4.460000038146973,
//              2.3399999141693115,
//            ],
//          },
//          'I': 4,
//          'L': {
//            'V': {
//              'DT': 'F32',
//              'D': [
//                19.809999465942383,
//                3.1700000762939453,
//                14.270000457763672,
//                0.05000000074505806,
//                -17.93000030517578,
//              ],
//            },
//            'I': 1,
//            'L': {
//              'S': {
//                'DT': 'F32',
//                'D': [
//                  [
//                    4.159999847412109,
//                    -4.71999979019165,
//                    -3.7100000381469727,
//                    -2.319999933242798,
//                    -13.699999809265137
//                  ],
//                  [
//                    4.079999923706055,
//                    -13.1899995803833,
//                    -5.710000038146973,
//                    18.559999465942383,
//                    -0.12999999523162842
//                  ],
//                  [
//                    7.630000114440918,
//                    -10.699999809265137,
//                    15.09000015258789,
//                    10.25,
//                    -18.15999984741211
//                  ],
//                ],
//              },
//            },
//            'R': {
//              'V': {
//                'DT': 'F32',
//                'D': [
//                  0.05000000074505806,
//                  9.739999771118164,
//                  7.079999923706055,
//                  15.489999771118164,
//                  -17.989999771118164,
//                ],
//              },
//              'I': 2,
//              'L': {
//                'S': {
//                  'DT': 'F32',
//                  'D': [
//                    [
//                      7.289999961853027,
//                      11.15999984741211,
//                      -9.510000228881836,
//                      -1.8899999856948853,
//                      -18.940000534057617
//                    ],
//                    [
//                      -9.630000114440918,
//                      18.81999969482422,
//                      -14.399999618530273,
//                      -1.909999966621399,
//                      -6.579999923706055
//                    ],
//                  ],
//                },
//              },
//              'R': {
//                'S': {
//                  'DT': 'F32',
//                  'D': [
//                    [
//                      3.430000066757202,
//                      10.90999984741211,
//                      11.619999885559082,
//                      -12.930000305175781,
//                      -11.65999984741211
//                    ],
//                    [
//                      19.809999465942383,
//                      3.1700000762939453,
//                      14.270000457763672,
//                      0.05000000074505806,
//                      -17.93000030517578
//                    ],
//                    [
//                      0.05000000074505806,
//                      9.739999771118164,
//                      7.079999923706055,
//                      15.489999771118164,
//                      -17.989999771118164
//                    ],
//                  ],
//                },
//              },
//            },
//          },
//          'R': {
//            'V': {
//              'DT': 'F32',
//              'D': [
//                2.7899999618530273,
//                -9.149999618530273,
//                6.559999942779541,
//                -18.59000015258789,
//                13.529999732971191,
//              ],
//            },
//            'I': 0,
//            'L': {
//              'V': {
//                'DT': 'F32',
//                'D': [
//                  -6.480000019073486,
//                  1.100000023841858,
//                  9.279999732971191,
//                  0.8999999761581421,
//                  6.090000152587891,
//                ]
//              },
//              'I': 1,
//              'L': {
//                'V': {
//                  'DT': 'F32',
//                  'D': [
//                    -15.970000267028809,
//                    -15.949999809265137,
//                    7.710000038146973,
//                    9.699999809265137,
//                    16.940000534057617,
//                  ]
//                },
//                'I': 2,
//                'L': {
//                  'S': {
//                    'DT': 'F32',
//                    'D': [
//                      [
//                        -9.880000114440918,
//                        -5.659999847412109,
//                        -16.149999618530273,
//                        4.460000038146973,
//                        2.3399999141693115,
//                      ],
//                    ],
//                  },
//                },
//                'R': {
//                  'S': {
//                    'DT': 'F32',
//                    'D': [
//                      [
//                        -8.130000114440918,
//                        -5.230000019073486,
//                        18.010000228881836,
//                        1.9700000286102295,
//                        9.079999923706055
//                      ],
//                      [
//                        -10.949999809265137,
//                        -19.579999923706055,
//                        9.050000190734863,
//                        17.389999389648438,
//                        3.299999952316284
//                      ],
//                      [
//                        -15.970000267028809,
//                        -15.949999809265137,
//                        7.710000038146973,
//                        9.699999809265137,
//                        16.940000534057617
//                      ],
//                    ],
//                  },
//                },
//              },
//              'R': {
//                'V': {
//                  'DT': 'F32',
//                  'D': [
//                    -6.480000019073486,
//                    1.100000023841858,
//                    9.279999732971191,
//                    0.8999999761581421,
//                    6.090000152587891
//                  ],
//                },
//                'I': 3,
//                'L': {
//                  'S': {
//                    'DT': 'F32',
//                    'D': [
//                      [
//                        -7.559999942779541,
//                        11.970000267028809,
//                        6.550000190734863,
//                        -7.539999961853027,
//                        15.899999618530273
//                      ],
//                      [
//                        -15.010000228881836,
//                        16.1200008392334,
//                        -10.420000076293945,
//                        -17.610000610351562,
//                        6.269999980926514
//                      ],
//                    ],
//                  },
//                },
//                'R': {
//                  'S': {
//                    'DT': 'F32',
//                    'D': [
//                      [
//                        -17.649999618530273,
//                        13.100000381469727,
//                        5.820000171661377,
//                        8.609999656677246,
//                        14.40999984741211
//                      ],
//                      [
//                        -6.480000019073486,
//                        1.100000023841858,
//                        9.279999732971191,
//                        0.8999999761581421,
//                        6.090000152587891
//                      ],
//                    ],
//                  },
//                },
//              },
//            },
//            'R': {
//              'V': {
//                'DT': 'F32',
//                'D': [
//                  19.40999984741211,
//                  -4.960000038146973,
//                  3.990000009536743,
//                  16.350000381469727,
//                  10.569999694824219,
//                ],
//              },
//              'I': 2,
//              'L': {
//                'S': {
//                  'DT': 'F32',
//                  'D': [
//                    [
//                      11.300000190734863,
//                      8.890000343322754,
//                      -17.65999984741211,
//                      -5.170000076293945,
//                      16.200000762939453
//                    ],
//                  ],
//                },
//              },
//              'R': {
//                'S': {
//                  'DT': 'F32',
//                  'D': [
//                    [
//                      19.40999984741211,
//                      -4.960000038146973,
//                      3.990000009536743,
//                      16.350000381469727,
//                      10.569999694824219
//                    ],
//                    [
//                      13.979999542236328,
//                      -8.210000038146973,
//                      17.010000228881836,
//                      -5.139999866485596,
//                      14.489999771118164
//                    ],
//                    [
//                      2.7899999618530273,
//                      -9.149999618530273,
//                      6.559999942779541,
//                      -18.59000015258789,
//                      13.529999732971191
//                    ]
//                  ],
//                },
//              },
//            },
//          },
//        },
//        'D': 'float32',
//      });
    });

    test(
        'should find the closest neighbours for [2.79, -9.15, 6.56, -18.59, 13.53]',
        () {
      final sample = Vector.fromList([2.79, -9.15, 6.56, -18.59, 13.53]);
      final result = kdTree.query(sample, 3).toList();


      expect((kdTree as KDTreeImpl).searchIterationCount, lessThanOrEqualTo(15));
      expect(result[0].pointIndex, 12);
      expect(result[1].pointIndex, 4);
      expect(result[2].pointIndex, 18);
      expect(result, hasLength(3));
    });

    test(
        'should find the closest neighbours for [13.98, -8.21, 17.01, -5.14, 14.49]',
        () {
      final sample = Vector.fromList([13.98, -8.21, 17.01, -5.14, 14.49]);
      final result = kdTree.query(sample, 4).toList();

      expect(
          (kdTree as KDTreeImpl).searchIterationCount, lessThanOrEqualTo(15));
      expect(result[0].pointIndex, 4);
      expect(result[1].pointIndex, 12);
      expect(result[2].pointIndex, 3);
      expect(result[3].pointIndex, 18);
      expect(result, hasLength(4));
    });

    test(
        'should find the closest neighbours for [-9.88, -5.66, -16.15, 4.46, 2.34]',
        () {
      final sample = Vector.fromList([-9.88, -5.66, -16.15, 4.46, 2.34]);
      final result = kdTree.query(sample, 4).toList();

      print(
          'Search iteration count: ${(kdTree as KDTreeImpl).searchIterationCount}');

      expect(result[0].pointIndex, 0);
      expect(result[1].pointIndex, 1);
      expect(result[2].pointIndex, 1);
      expect(result[3].pointIndex, 1);
      expect(result, hasLength(4));
    });

    test(
        'should find the closest neighbours for [-9.88, -5.66, -16.15, 4.46, 2.34]',
        () {
      final sample = Vector.fromList([-9.88, -5.66, -16.15, 4.46, 2.34]);
      final result = kdTree.query(sample, 20).toList();

      print(
          'Search iteration count: ${(kdTree as KDTreeImpl).searchIterationCount}');

      expect(result[0].pointIndex, 19);
      expect(result, hasLength(20));
    });
  });
}
