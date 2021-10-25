## Patientory Team B

**Kidney Disease and Fitness Data Project**

> Prerequisites:

	 - [ ] Python (Anaconda distribution preferred)
	 - [ ] pandas
	 - [ ] numpy
	 - [ ] matplotlib
	 - [ ] statsmodels
	 - [ ] sklearn
	 - [ ] Tableau for Fitness Data
	 - [ ] Postgres Database

> Steps to implement the model - Kidney Disease:

 1. Setup a Postgres database server 
	 https://www.postgresql.org/download/  
	 `psql -h <hostname> -p <port> -U <username> -W <password>`
	 
	 `create database kidney;`
 2. Import the postgres database export using `psql -U postgres kidney < kidney.pgsql`
	 https://www.dropbox.com/s/7hjzt4gaeh49jvv/kidney.pgsql?dl=0
 3. Provide connection details and credentials in notebook section "Connecting to PostgreSQL"
 4. Run the notebook `./Kidney_Disease/Code/synthea_patientory_kidney_disease.ipynb` for results.


> Steps to implement fitness data exploration

1. Import the postgres database export using:

 	 https://www.dropbox.com/s/2kb6hyhns44gz5q/fitness.pgsql?dl=0
	 
	`create database fitness;`
	
	`psql -U postgres fitness < fitness.pgsql`
	
2. Visualization are placed in a Tableau Workbook at `./Fitness/Code/fitness_data_dashboard.twbx`
3. For additional data processing and xml parsing the notebooks at the below location can be executed `./Fitness/Code/fitness_data_processing.ipynb `and` ./Fitness/Code/fitness_data_xml_parse.ipynb`
4.  Provide connection details and credentials in notebook section "Connecting to PostgreSQL"
