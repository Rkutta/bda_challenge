from pyspark import SparkContext
import datetime
import csv
import functools
import json
import numpy as np
import sys
 
def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]
    CAT_CODES = {'452210','452311','455120','722410','722511','722513','446110','446191','311811','722515','455210','445220','445230','445291','445292','445299','445110'}
    CAT_GROUP = {'452210':0,'452311':0,'445120':1,'722410':2,'722511':3,'722513':4,'446110':5,'446191':5,'311811':6,'722515':6,'445210':7,'445220':7,'445230':7,'445291':7,'445292':7,'445299':7,'445110':8}
    # Filter places of interest
    def filterPOIs(_, lines):
        reader = csv.reader(lines)
        for line in reader:
            if line[9] in CAT_CODES:
                yield (line[0], int(CAT_GROUP[line[9]]))
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
            .cache()    
    # Compute number of stores per group
    storeGroup = dict(rddD.collect())
    groupCount = rddD. \
        map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y).map(lambda x: x[1]) \
        .collect()
    # Filter pattern data, explode visits by day
    def extractVisits(storeGroup, _, lines):
        reader = csv.reader(lines)
        for line in reader:
            if (('2019' in line[12]) or ('2020' in line[12])) and (line[0] in list(storeGroup.keys())):
                start_day = datetime.datetime.strptime(line[12][:10], '%Y-%m-%d')
                visits = json.loads(line[16])
                base_day = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
                for i in range(len(visits)):
                    day = start_day + datetime.timedelta(days=i)
                    days = day - base_day
                    yield ((storeGroup[line[0]],days.days), visits[i])
    
    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup)) 
    # Compute daily stats for each group and convert to CSV format
    def computeStats(groupCount, _, records):
        for record in records:
            lst = sorted(list(record[1]))
            median = lst[int(len(lst)/2)]
            var = 0
            for visits in lst:
                var += (visits - median)**2
            var = var / groupCount[record[0][0]-1]
            std = var**0.5
            low = max(median - std, 0)
            high = max(median + std, 0)
            base_day = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
            day = base_day + datetime.timedelta(days=record[0][1])
            year = day.strftime('%Y')
            day_str = day.strftime('%Y-%m-%d')
            yield (record[0][0], year+','+day_str+','+str(median)+','+str(low)+','+str(high))
    
    rddI = rddG.groupByKey() \
        .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))
    
    # Sort data for output
    rddJ = rddI.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()
    
    # Write data output
    # Remove the output folder if it's already there
    #!rm -rf /content/output/*
    
    filenames = ['big_box_grocers','convenience_stores','drinking_places','full_service_restaurants','limited_service_restaurants','pharmacies_and_drug_stores','snack_and_retail_bakeries','specialty_food_stores','supermarkets_except_convenience_stores']
    i = 0
    for filename in filenames:
        rddJ.filter(lambda x: x[0]==i or x[0]==-1).values() \
            .saveAsTextFile(f'{OUTPUT_PREFIX}/{filename}')
        i += 1    
    
    
if __name__=='__main__':
    sc = SparkContext()
    main(sc)
