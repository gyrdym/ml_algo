// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'dart:convert';
import 'dart:io';

import 'package:collection/collection.dart';
import 'package:ml_algo/kd_tree.dart';

import 'models.dart';

List<Ticket> parseData(List<dynamic> data) {
  final tickets = <Ticket>[];
  for (var entry in data) {
    final ticketW = TicketWrapper.fromMap(entry as Map<String, dynamic>);
    tickets.add(ticketW.ticket);
  }
  return tickets;
}

List<Ticket> removeContactsTickets(List<Ticket> data, num selectedContactId) =>
    data
        .where((ticket) => ticket.contactId != selectedContactId)
        .toList(growable: false);

/// this make a flat table of all items, one item per column
List<Iterable<num>> prepareData(
    List<Ticket> filteredTickets, Set<num> articleIdsSet) {
  // populate a set of article ids

  final vectors = <Iterable<num>>[];
  for (final ticket in filteredTickets) {
    final vectorRow = <num>[];
    for (final articleId in articleIdsSet) {
      if (ticket.items.any((i) => i.articleId == articleId)) {
        final itemWeighted =
            ticket.items.firstWhere((i) => i.articleId == articleId).qt;
        vectorRow.add(itemWeighted);
      } else {
        vectorRow.add(0);
      }
    }
    vectors.add(vectorRow);
  }
  return vectors;
}

List<num> buildIndex(List<Ticket> tickets, {String path = indexPath}) {
  final index = <num>[];
  for (final ticket in tickets) {
    // ignore: unused_local_variable
    for (final item in ticket.items) {
      index.add(ticket.timestamp);
    }
  }
  return index;
}

const String dataPath = 'example/pos/input/ticket_917_known_contacts.json';
const String indexPath = 'example/pos/output/index.txt';
const String vectorsPath = 'example/pos/output/vectors.txt';
const String selectedContactIndexPath =
    'example/pos/output/index_selected_contact.txt';
const String selectedContactVectorsPath =
    'example/pos/output/vectors_selected_contact.txt';

Set<num> itemsIds(List<Ticket> tickets) {
  final articleIdsSet = <double>{};
  for (final ticket in tickets) {
    for (final item in ticket.items) {
      articleIdsSet.add(item.articleId);
    }
  }
  articleIdsSet.sorted((a, b) => a.compareTo(b));
  return articleIdsSet;
}

void main(List<String>? args) async {
  var selectedContactId = 1.0;
  if (args == null || args.isEmpty || args.first.isEmpty) {
    print('default selectedContactId = 1.0');
  } else {
    if (double.tryParse(args.first) == null) {
      throw 'provide a valid id';
    }
    if (availableContactIds.any((id) => id == double.parse(args.first)) ==
        false) {
      throw 'provide an id included in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57]';
    }
  }
  final file = File(dataPath);
  final data = json.decode(await file.readAsString()) as List<dynamic>;
  final ticketsFull = parseData(data);
  final itemsIdsSet = itemsIds(ticketsFull);

  /// ----- select the contact's tickets -----
  // we select a contactId
  // and look for its tickets
  // tickets belonging to this contact will not be included in the queried data

  final contactTickets = <Ticket>[];
  for (final ticket in ticketsFull) {
    if (ticket.contactId == selectedContactId) {
      contactTickets.add(ticket);
    }
  }
  // save the contactTickets' index for archive
  final selectedContactIndex =
      buildIndex(contactTickets, path: selectedContactIndexPath);
  File(selectedContactIndexPath)
      .writeAsStringSync(selectedContactIndex.join('\n'));
  // save the contactTickets' vectors for archive
  final vectorsSelectedContact = prepareData(contactTickets, itemsIdsSet);
  File(selectedContactVectorsPath).writeAsStringSync(
      vectorsSelectedContact.map((v) => v.join(',')).join('\n'));

  final filteredTickets = removeContactsTickets(ticketsFull, selectedContactId);

  /// build and save index for archive
  final index = buildIndex(filteredTickets);
  File(indexPath).writeAsStringSync(index.join('\n'));

  // build and save the vectors for archive
  final vectors = prepareData(filteredTickets, itemsIdsSet);
  File(vectorsPath)
      .writeAsStringSync(vectors.map((v) => v.join(',')).join('\n'));

  final tree = KDTree.fromIterable(vectors);
  final neighbourCount = 3;
  final neighboursTimestamps = <num>[];
  // select all vectors from selectedContact's ticket.items
  print("select all vectors from selectedContact's ticket.items");
  for (final vectorSelected in vectorsSelectedContact) {
    final neighbours = tree.queryIterable(vectorSelected, neighbourCount);
    print(neighbours);
    for (final neighbour in neighbours) {
      neighboursTimestamps.add(index[neighbour.index]);
    }
  }

  // find each neighbour tickets' full content
  final neighboursTickets = <Ticket>[];
  final neighboursContactIds = <num>[];
  for (final filteredTicket in filteredTickets) {
    if (neighboursTimestamps.any((nt) => nt == filteredTicket.timestamp)) {
      neighboursTickets.add(filteredTicket);
      neighboursContactIds.add(filteredTicket.contactId);
    }
  }

  /// finding the neighbours ticket id in index (timestamp)
  final top3 = top3Neighbours(neighboursContactIds);

  print('Nearest: ${top3[0]}');
  print('Second nearest: ${top3[1]}');
  print('Third nearest: ${top3[2]}');

// for information, double check
/*   for (final neighbourTicket in neighboursTickets) {
    print(neighbourTicket.toString());
  } */
}

List<num> top3Neighbours(List<num> ids) {
  // Create a map to count occurrences
  Map<num, num> frequencyMap = {};
  for (final number in ids) {
    if (frequencyMap.containsKey(number)) {
      frequencyMap[number] = frequencyMap[number]! + 1;
    } else {
      frequencyMap[number] = 1;
    }
  }

  // Convert map to a list of entries and sort by value (frequency) in descending order
  List<MapEntry<num, num>> sortedEntries = frequencyMap.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));

  // Get the top 3 most frequent ids
  List<num> top3FrequentNumbers =
      sortedEntries.take(3).map((e) => e.key).toList();

  return top3FrequentNumbers;
}
