from collections import deque
from typing import Dict, List
import pandas as pd
from Infra import Infra, SECTION_RESULT

class InfraManager():
    def __init__(self, config, name="Static Control", simMode=True, filenames=None, isExternal=False):
        self.sigTypeName = name
        self.config = config
        self.filenames = filenames
        self.isExternalSignal = isExternal
        self.preinit()
        #init Infra
        self._Infra: List = self._make_Infra()

    def preinit(self):
        pass

    def _make_Infra(self) -> List[Infra]:
        pass

    def __str__(self):
        return self.sigTypeName

    def getInfras(self):
        return self._Infra

    def getInfra(self):
        if len(self._Infra) > 0:
            return self._Infra[-1]
        else:
            return None

    def extract_excel(self, saveCompare=False):
        section_results = deque()
        append_result = section_results.append
        file_name = ""
        if saveCompare is False:
            data = self.getInfra()
        else:
            data = self.compareInfra
            file_name = 'extract_'

        timedata = data.getSections()['0'].getDatabyID(SECTION_RESULT.TIME)
        for i, time in enumerate(timedata):
            for section_id, section in data.getSections().items():
                section_co2_emission, section_volume, traffic_queue, green_time = section.collect_data()

                # print("%s - v: %d, Q: %d"%(section_id, section_volume, section_queue))
                append_result({
                    'Time': time,
                    'Section': section_id,
                    'Section_CO2_Emission': section.getDatabyID(SECTION_RESULT.SECTION_CO2)[i],
                    'Section_Volume': section.getDatabyID(SECTION_RESULT.VOLUME)[i],
                    'traffic_queue': section.getDatabyID(SECTION_RESULT.TRAFFIC_QUEUE)[i],
                    'green_time': section.getDatabyID(SECTION_RESULT.GREEN_TIME)[i],
                    'sectionBound': str(section.direction)
                })

        df = pd.DataFrame(section_results)

        co2_emission_df = df.pivot(index='Time', columns='Section', values='Section_CO2_Emission')
        volume_df = df.pivot(index='Time', columns='Section', values='Section_Volume')
        queue_df = df.pivot(index='Time', columns='Section', values='traffic_queue')
        greentime_df = df.pivot(index='Time', columns='Section', values='green_time')
        with pd.ExcelWriter(file_name+'section_results.xlsx') as writer:
            co2_emission_df.to_excel(writer, sheet_name='Section_CO2_Emission')
            volume_df.to_excel(writer, sheet_name='Section_Volume')
            queue_df.to_excel(writer, sheet_name='traffic_queue')
            greentime_df.to_excel(writer, sheet_name='traffic_queue')

        print("Maked Excel")