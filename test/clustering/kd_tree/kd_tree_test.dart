import 'package:ml_algo/src/clustering/kd_tree/kd_tree.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('KDTree', () {
    final leafSize = 2;
    final samples = DataFrame([
      [1, 0.3, 3, 4, 11],
      [9.67, 0.5, 7, 4, 3],
      [19, 2, 1, 4, 6],
      [60, 2, 2, 3, 3],
      [33.12, 10, 4, 2, 7],
      [-100, -1, 3, 2, 6],
    ], headerExists: false);
    // 1)
    // variances:
    // 1375.9759 15.52 4.2666667 0.96666667 8.8
    // choose 1st column, splitting value 3.79833:
    //
    // root: {
    //   C: [
    //     {
    //        C: [],
    //        V: 3.79833,
    //        I: 0,
    //        P: "LT",
    //        S: [
    //          [1,     0.3,  3,  4, 11],
    //          [ -100,  -1,  3,  2,  6],
    //        ],
    //      },
    //      {
    //        C: [],
    //        V: 3.79833,
    //        I: 0,
    //        P: "GT",
    //        S: [
    //          [9.67,  0.5,  7,  4,  3],
    //          [19,      2,  1,  4,  6],
    //          [60,      2,  2,  3,  3],
    //          [33.12,  10,  4,  2,  7],
    //        ]
    //      }
    //   ]
    // }
    //
    // 2)
    // variances (right node):
    // 481.08076 18.5625 7 0.91666667 4.25
    // choose 1st column, splitting value 30.4474
    //
    // root: {
    //   C: [
    //     {
    //        C: [],
    //        V: 3.79833,
    //        I: 0,
    //        P: "LT",
    //        S: [
    //          [1,     0.3,  3,  4, 11],
    //          [ -100,  -1,  3,  2,  6],
    //        ],
    //      },
    //      {
    //        C: [
    //          {
    //            C: [],
    //            V: 30.4474,
    //            I: 0,
    //            P: "LT",
    //            S: [
    //              [9.67,  0.5,  7,  4,  3],
    //              [19,      2,  1,  4,  6],
    //            ],
    //          },
    //          {
    //            C: [],
    //            V: 30.4474,
    //            I: 0,
    //            P: "GT",
    //            S: [
    //              [60,      2,  2,  3,  3],
    //              [33.12,  10,  4,  2,  7],
    //            ],
    //          },
    //        ],
    //        V: 3.79833,
    //        I: 0,
    //        P: "GT",
    //      }
    //   ]
    // }
    final tree = KDTree(samples, leafSize: leafSize);

    test('should build a proper tree', () {
      expect(tree.toJson(), {
        'L': 2,
        'D': 'float32',
        'R': {
          'C': [
            {
              'C': [],
              'S': {
                'DT': 'F32',
                'D': [
                  [1, 0.30000001192092896, 3, 4, 11],
                  [-100, -1, 3, 2, 6],
                ],
              },
              'V': 3.798332850138346,
              'I': 0,
              'P': 'LT',
              'L': 1,
            },
            {
              'C': [
                {
                  'C': [],
                  'S': {
                    'DT': 'F32',
                    'D': [
                      [9.670000076293945, 0.5, 7, 4, 3],
                      [19, 2, 1, 4, 6],
                    ],
                  },
                  'P': 'LT',
                  'V': 30.447499752044678,
                  'I': 0,
                  'L': 2,
                },
                {
                  'C': [],
                  'S': {
                    'DT': 'F32',
                    'D': [
                      [60, 2, 2, 3, 3],
                      [33.119998931884766, 10, 4, 2, 7],
                    ],
                  },
                  'P': 'GET',
                  'V': 30.447499752044678,
                  'I': 0,
                  'L': 2,
                },
              ],
              'P': 'GET',
              'V': 3.798332850138346,
              'I': 0,
              'L': 1,
            }
          ],
          'L': 0,
        }
      });
    });

    test('should persist leaf size parameter', () {
      expect(tree.leafSize, leafSize);
    });

    test('should persist dtype parameter', () {
      expect(tree.dtype, DType.float32);
    });
  });
}
