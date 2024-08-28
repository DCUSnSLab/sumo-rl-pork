import os
import pickle
from typing import Dict, List

import traci
from inframanager import InfraManager
from Infra import SDetector, SStation, SSection, Infra, SECTION_RESULT

class RunSimulation(InfraManager):
    def __init__(self, config, name="Static Control", isExternalSignal=False):
        super().__init__(config, name, simMode=True, filenames=None, isExternal=isExternalSignal)
        self.stepbySec = 1
        self.colDuration = 30  # seconds

        self.traffic_light_id = "TLS_0"
        self.isStop = True
        self.step = 0

        self.original_logic = None
        self.logic = None
        self._rtinfra = self.getInfra()

    def preinit(self):
        self.__set_SUMO()

    def _make_Infra(self) -> List[Infra]:
        infra = None
        DetectorClass = None
        StationClass = None
        SectionClass = None

        DetectorClass = SDetector
        StationClass = SStation
        SectionClass = SSection

        dets = self.__init_detector(DetectorClass)
        station_objects = self.__init_station(dets, StationClass)
        section_objects = self.__init_section(station_objects, SectionClass)

        return [Infra(self.config.sumocfg_path, self.config.scenario_path, self.config.scenario_file, section_objects, self.sigTypeName)]

    def __get_detector_ids(self, config):
        detector_ids = []
        with open(os.path.join(config.scenario_path, config.scenario_file), "r") as f:
            for line in f:
                if "inductionLoop" in line:
                    parts = line.split('"')
                    detector_ids.append(parts[1])
        return detector_ids

    def __init_detector(self, detectorclass=SDetector):
        return [detectorclass(detector_id) for detector_id in self.__get_detector_ids(self.config)]

    def __init_station(self, dets, stationclass=SStation):
        station_objects = {}
        for detector in dets:
            if detector.station_id not in station_objects:
                station_objects[detector.station_id] = stationclass(detector.station_id)
            station_objects[detector.station_id].addDetector(detector)
        return station_objects

    def __init_section(self, stations, sectionclass=SSection) -> Dict[int, SSection]:
        section_objects = {}
        logic = None
        print(self.isExternalSignal)
        if self.isExternalSignal is False and sectionclass is SSection:
            logic = traci.trafficlight.getAllProgramLogics("TLS_0")[0]

        for station_id in stations:
            section_id = station_id[1]
            if section_id not in section_objects:
                section_objects[section_id] = sectionclass(section_id)
            section_objects[section_id].addStation(stations[station_id])

        #set Default greentime
        for sid, section in section_objects.items():
            if logic is not None:
                section.default_greentime = logic.phases[section.direction.value[1]].duration
        return section_objects

    def __set_SUMO(self):
        traci.start(["sumo-gui", "-c", self.config.sumocfg_path, "--start", "--quit-on-end"])
        traci.simulationStep()

    def terminate(self):
        self.isStop = True

    def isTermiated(self):
        return self.isStop

    def saveData(self, filename):
        if self.isStop is True:
            print('save data clicked')
            with open(self._rtinfra.setSaveFileName(filename), "wb") as f:
                pickle.dump(self._rtinfra, f)
                print('---file saved at ',self._rtinfra.getFileName())
            #self.extract_excel()

    def _refreshSignalPhase(self):
        traci.trafficlight.setProgramLogic("TLS_0", self.logic)

    def _signalControl(self):
        pass

    def run_simulation(self):
        print('---- start Simulation (signController : ',self.sigTypeName, ") ----")
        self.step = 0
        self.isStop = False

        while not self.isStop and self.step <= 11700:
            #start_time = time.time()
            traci.simulationStep()

            #set logic every step
            self.logic = traci.trafficlight.getAllProgramLogics("TLS_0")[0]

            self._signalControl()
            if self.sigTypeName != "Reinforement Learning based Control":
                self._refreshSignalPhase()
            # print('Green times: ', end='')

            self._rtinfra.update()

            self.step += 1

        self.isStop = True
        traci.close()


    def Check_TrafficLight_State(self):
        try:
            signal_states = traci.trafficlight.getRedYellowGreenState("TLS_0")
        except traci.exceptions.TraCIException:
            signal_states = 'N/A'
        return signal_states