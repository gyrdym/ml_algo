void validateClassLabels(num positiveLabel, num negativeLabel) {
  if (positiveLabel == negativeLabel) {
    throw Exception('Positive and negatve labels must be different, both '
        'labels are equal to $positiveLabel instead');
  }
}
