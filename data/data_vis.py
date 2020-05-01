import csv 
  
filename = "aapl.csv"
  
fields = [] 
rows = [] 
  
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
      
    fields = next(csvreader) 
  
    for row in csvreader: 
        rows.append(row) 
  
    print("Total no. of rows: %d"%(csvreader.line_num)) 
  
