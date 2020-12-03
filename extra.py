def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--countries',
        action='store',
        dest='countries',
        help='Countries on CSV format. ' +
        'It must exact match the data names or you will get out of bonds error.',
        metavar='COUNTRY_CSV',
        type=str,
        default="")
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        dest='download_data',
        help='Download fresh data and then run',
        default=False
    )

    parser.add_argument(
        '--start-date',
        required=False,
        action='store',
        dest='start_date',
        help='Start date on MM/DD/YY format ... I know ...' +
        'It defaults to first data available 1/22/20',
        metavar='START_DATE',
        type=str,
        default="1/22/20")

    parser.add_argument(
        '--prediction-days',
        required=False,
        dest='predict_range',
        help='Days to predict with the model. Defaults to 150',
        metavar='PREDICT_RANGE',
        type=int,
        default=150)

    parser.add_argument(
        '--S_0',
        required=False,
        dest='s_0',
        help='S_0. Defaults to 100000',
        metavar='S_0',
        type=int,
        default=100000)

    parser.add_argument(
        '--I_0',
        required=False,
        dest='i_0',
        help='I_0. Defaults to 2',
        metavar='I_0',
        type=int,
        default=2)

    parser.add_argument(
        '--R_0',
        required=False,
        dest='r_0',
        help='R_0. Defaults to 0',
        metavar='R_0',
        type=int,
        default=10)

    args = parser.parse_args()

    country_list = []
    if args.countries != "":
        try:
            countries_raw = args.countries
            country_list = countries_raw.split(",")
        except Exception:
            sys.exit("QUIT: countries parameter is not on CSV format")
    else:
        sys.exit("QUIT: You must pass a country list on CSV format.")

    return (country_list, args.download_data, args.start_date, args.predict_range, args.s_0, args.i_0, args.r_0)


def sumCases_province(input_file, output_file):
    with open(input_file, "r") as read_obj, open(output_file,'w',newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
               
        lines=[]
        for line in csv_reader:
            lines.append(line)    

        i=0
        ix=0
        for i in range(0,len(lines[:])-1):
            if lines[i][1]==lines[i+1][1]:
                if ix==0:
                    ix=i
                lines[ix][4:] = np.asfarray(lines[ix][4:],float)+np.asfarray(lines[i+1][4:] ,float)
            else:
                if not ix==0:
                    lines[ix][0]=""
                    csv_writer.writerow(lines[ix])
                    ix=0
                else:
                    csv_writer.writerow(lines[i])
            i+=1    


def download_data(url_dictionary):
    #Lets download the files
    for url_title in url_dictionary.keys():
        urllib.request.urlretrieve(url_dictionary[url_title], "./data/" + url_title)


def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)




r_0 = pmtros['r_0_def']
i_0 = pmtros['I_0_def']
s_0 = pmtros['S_0_def']
predict_range = pmtros['PREDICT_RANGE_def']
startdate = pmtros['START_DATE_def']
countries = pmtros['COUNTRY_CSV_def']

pmtros = {
    'S_0_def': 100000,
    'I_0_def': 2,
    'R_0_def': 10,
    'PREDICT_RANGE_def': 150,
    'START_DATE_def': "2020-01-22",
    'COUNTRY_CSV_def': ['BRAISL'],
    'csv_confirmed': 'time_series_covid19_confirmed_global.csv',
    'csv_death': 'time_series_covid19_deaths_global.csv',
    'csv_recovered': 'time_series_covid19_recovered_global.csv'
}


def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, country, s_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1))


    def train(self):
        recovered = self.load_recovered(self.country)
        death = self.load_dead(self.country)
        data = (self.load_confirmed(self.country) - recovered - death)
        
        optimal = minimize(loss, [0.001, 0.001], args=(data, recovered, self.s_0, self.i_0, self.r_0), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, data, recovered, death, self.country, self.s_0, self.i_0, self.r_0)
        df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{self.country}.png")