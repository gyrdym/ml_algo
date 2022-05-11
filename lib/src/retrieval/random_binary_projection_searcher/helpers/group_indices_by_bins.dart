Map<int, List<int>> groupIndicesByBins(Iterable<int> binIds) {
  var i = 0;

  return binIds.fold<Map<int, List<int>>>(<int, List<int>>{},
      (previousValue, binId) {
    previousValue.putIfAbsent(binId, () => []);

    return previousValue..update(binId, (value) => value..add(i++));
  });
}
