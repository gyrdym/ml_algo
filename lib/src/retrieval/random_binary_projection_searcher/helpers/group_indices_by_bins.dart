Map<num, List<int>> groupIndicesByBins(Iterable<num> binIds) {
  var i = 0;

  return binIds.fold<Map<num, List<int>>>(<num, List<int>>{},
      (previousValue, binId) {
    previousValue.putIfAbsent(binId, () => []);

    return previousValue..update(binId, (value) => value..add(i++));
  });
}
