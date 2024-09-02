from datetime import datetime
from collections import deque
from enum import Enum
from typing import Dict

import traci
import math
from traci import TraCIException

class Config_SUMO:
    # SUMO Configuration File
    sumocfg_path = "New_TestWay/test_cfg.sumocfg"
    # SUMO Scenario File Path
    scenario_path = ""
    # SUMO Scenario File(.add.xml)
    scenario_file = "test.net_mergy.xml"
    route_file = "generated_flows_pm.xml"

    max_green = 120

    sumoBinary = r'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui'

class Direction(Enum):
    SB = (0, 4)
    NB = (1, 6)
    EB = (2, 0)
    WB = (3, 2)

    @classmethod
    def from_first_value(cls, first_value):
        for member in cls:
            if member.value[0] == first_value:
                return member
        raise ValueError(f"{first_value} is not a valid first value for {cls.__name__}")

class InputStation(Enum):
    SB = '000000'
    NB = '010021'
    EB = '020018'
    WB = '030017'

class SMUtil:
    MPStoKPH = 3.6
    secPerHour = 3600
    sec = 1
    interval = 10

class SECTION_RESULT(Enum):
    TIME = 'Time'
    TIMEINT = 'Time Interval'
    SECTIONID = 'Sectionid'
    SECTION_CO2 = 'Section_CO2_Emission'
    VOLUME = 'Section_Volume'
    SPEED_INT = 'Section Speed by Interval'
    TRAFFIC_QUEUE = 'traffic_queue'
    GREEN_TIME = 'green_time'
    DIRECTION = 'direction'

class TOTAL_RESULT(Enum):
    TIME = 'Time'
    TOTAL_CO2 = 'Total CO2'
    TOTAL_CO2_ACC = 'Total CO2 ACC'
    TOTAL_VOLUME = 'TOtal Volume'

    @classmethod
    def from_string(cls, string_value):
        for mode in cls:
            if mode.value == string_value:
                return mode
        raise ValueError(f"{string_value} is not a valid SignalMode value")

def get_input_station_value(direction: Direction) -> str:
    # Direction의 name으로 InputStation을 찾아서 value를 반환
    return InputStation[direction.name].value

# Detector
class Detector:
    def __init__(self, id):
        self.id = id
        self.aux, self.bound, self.station_id, self.detector_id = self.parse_detector_id(id)
        self.flow = 0
        self.density = 0
        self.volumes = deque()
        self.speeds = deque()
        self.append_volumes = self.volumes.append
        self.append_speeds = self.speeds.append
        self.prevVehicles = tuple()

    def __str__(self):
        return f"Detector {self.id} at station {self.station_id} volumes {len(self.volumes)} and speeds {len(self.speeds)}"

    def __repr__(self):
        return f""
    def parse_detector_id(self, id):
        parts = id.split('_')
        if len(parts) != 2 or not parts[0].startswith("Det"):
            raise ValueError(f"Invalid detector ID format: {id}")
        det_info = parts[1]
        aux = det_info[0]
        bound = Direction.from_first_value(int(det_info[1]))
        station_id = det_info[0:6]
        detector_id = det_info[6:]
        return aux, bound, station_id, detector_id

    #update detection data by interval
    def update(self):
        pass

    def getVolume(self):
        if len(self.volumes) > 0:
            return self.volumes[-1]
        else:
            return -1

    def getVehicles(self):
        return self.prevVehicles

    def getSpeed(self):
        if len(self.speeds) > 0:
            return self.speeds[-1]
        else:
            return -1

        # 직렬화할 데이터를 정의하는 메서드
    def __getstate__(self):
        # 기본 상태를 가져온 후, flow, density와 parse_detector_id()에서 생성된 필드를 제거
        state = self.__dict__.copy()
        del state['flow']
        del state['density']
        del state['aux']
        del state['bound']
        del state['station_id']
        del state['detector_id']
        state['__class__'] = Detector
        return state

    # 역직렬화된 데이터를 객체 상태에 복원하는 메서드
    def __setstate__(self, state):
        # 상태를 설정하고, flow와 density, 그리고 parse_detector_id()의 필드를 다시 설정
        self.__dict__.update(state)
        self.flow = 0
        self.density = 0
        self.aux, self.bound, self.station_id, self.detector_id = self.parse_detector_id(self.id)
        self.__class__ = DDetector #state.pop('__class__', Detector)

class SDetector(Detector):
    def __int__(self, id):
        super().__init__(id)


    def update(self):
        vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(self.id)
        #check duplicated vehicles
        dupvol = 0
        speedcnt = 0
        volume = 0
        speed = 0
        for veh in vehicle_ids:
            if self.prevVehicles is not None and veh in self.prevVehicles:
                dupvol += 1
            else:
                speed +=  traci.inductionloop.getIntervalMeanSpeed(self.id)
                speedcnt += 1
            # if self.id == 'Det_02000000' or self.id == 'Det_02000001' or self.id == 'Det_12002604':
            #     print(" --- each %s %s -> u : %d"%(self.id, veh, self.speed))



        volume = traci.inductionloop.getLastStepVehicleNumber(self.id) - dupvol
        self.flow = volume * SMUtil.secPerHour / SMUtil.sec
        speed = 0 if speedcnt == 0 else speed / speedcnt
        self.density = 0 if speed == 0 else self.flow / (speed * SMUtil.MPStoKPH)
        # if self.id == 'Det_02000000' or self.id == 'Det_02000001' or self.id == 'Det_12002604':
        #     print("%s -> v : %d, u : %d, k : %d" % (self.id, self.volume, self.speed, self.density))
            #print('--- lsvid : ', vehicle_ids, self.prevVehicles, traci.inductionloop.getLastStepVehicleNumber(self.id), dupvol )
        self.prevVehicles = vehicle_ids
        self.append_volumes(volume)
        self.append_speeds(speed)

class DDetector(Detector):
    def __int__(self, id):
        super().__init__(id)

    def update(self):
        print('test')

class Station:
    def __init__(self, id, detectors=None):
        self.id = id
        self.dets = [] if detectors is None else detectors
        self.direction = None

        self.volumes = deque()
        self.speeds = deque()
        self.speeds_int = deque()
        self.exitVolumes = deque()
        self.append_volumes = self.volumes.append
        self.append_exitVolume = self.exitVolumes.append
        self.append_speeds = self.speeds.append
        self.append_speeds_int = self.speeds_int.append

        self.__define_direction()

    def __define_direction(self):
        if not hasattr(self, 'direction') or self.direction is None:
            self.direction = None if len(self.dets) == 0 else self.dets[0].bound

    def addDetector(self, detector):
        self.dets.append(detector)
        self.__define_direction()

    def update(self):
        pass

    def getVolume(self):
        if len(self.volumes) > 0:
            return self.volumes[-1]
        else:
            return -1

    def getSpeed(self):
        if len(self.speeds) > 0:
            return self.speeds[-1]
        else:
            return -1

    def getSpeedInt(self):
        if len(self.speeds_int) > 0:
            return self.speeds_int[-1]
        else:
            return -1

    def getSpeedInts(self):
        return self.speeds_int

    def getExitVolume(self):
        if len(self.exitVolumes) > 0:
            return self.exitVolumes[-1]
        else:
            return -1

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['direction']
        state['__class__'] = Station
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__define_direction()
        self.__class__ = DStation #state.pop('__class__', Station)
        #self.direction = None if len(self.dets) == 0 else self.dets[0].bound

class SStation(Station):
    def __init__(self, id, detectors=None):
        super().__init__(id, detectors)
        self.inputVeh = set()
        self.exitVeh = set()

    def update(self):
        volume = 0
        speed = 0
        exitVolume = 0
        self.inputVeh = set()
        self.exitVeh = set()

        for det in self.dets:
            det.update()

            if det.aux == '1':
                exitVolume += det.getVolume()
                self.exitVeh.update(det.getVehicles())
            else:
                volume += det.getVolume()
                speed += det.getSpeed()
                self.inputVeh.update(det.getVehicles())

        # if self.id == '020018' or self.id == '020018':
        #     print('--station id',self.id, self.inputVeh)

        speed = -1 if volume == 0 else speed / volume
        volume = volume if volume == 0 or volume < len(self.inputVeh) else len(self.inputVeh)
        exitVolume = exitVolume if exitVolume == 0 or exitVolume < len(self.exitVeh) else len(self.exitVeh)

        self.append_volumes(volume)
        self.append_speeds(speed)
        self.append_exitVolume(exitVolume)

        # calculate speed_int
        scnt = len(self.speeds)
        if scnt != 0 and scnt % SMUtil.interval == 0:
            last_interval_speeds = list(self.speeds)[-SMUtil.interval:]
            valid_speeds = [s for s in last_interval_speeds if s != -1]

            # 유효한 속도의 개수
            valid_count = len(valid_speeds)
            average_speed = sum(valid_speeds) / valid_count if valid_count > 0 else 0
            self.append_speeds_int(average_speed)
        # if self.id == '020018':
        #     print('station id',self.id,', volume: ',self.getVolume(), ' speed: ',len(self.speeds_int), (self.getSpeedInt() * SMUtil.MPStoKPH))
        # #     #print('station id : ', self.id, 'iv: ',self.inputVeh, 'ev: ', self.exitVeh)

    def getVehicleData(self):
        return list(self.inputVeh), list(self.exitVeh)

    def getInputVehIds(self):
        return self.inputVeh

    def getExitVehIds(self):
        return self.exitVeh

class DStation(Station):
    def __init__(self, id, detectors=None):
        super().__init__(id, detectors)

    def update(self):
        pass

class Section:
    def __init__(self, id, stations):
        self.id = id
        self.stations = [] if stations is None else stations
        self.direction = None
        self.default_greentime = 0

        #append data
        self.__time = deque()
        self.__timeint = deque()
        self.__section_co2 = deque()
        self.__section_volumes = deque()
        self.__section_speedint = deque()
        self.__section_queues = deque()
        self._section_greentime = deque()
        self.append_section_time = self.__time.append
        self.append_section_timeint = self.__timeint.append
        self.append_section_co2 = self.__section_co2.append
        self.append_section_volumes = self.__section_volumes.append
        self.append_section_speedint = self.__section_speedint.append
        self.append_section_queues = self.__section_queues.append
        self.append_section_greentime = self._section_greentime.append

        self.dataDic = dict()
        self.dataDic[SECTION_RESULT.TIME] = self.__time
        self.dataDic[SECTION_RESULT.TIMEINT] = self.__timeint
        self.dataDic[SECTION_RESULT.SECTION_CO2] = self.__section_co2
        self.dataDic[SECTION_RESULT.TRAFFIC_QUEUE] = self.__section_queues
        self.dataDic[SECTION_RESULT.GREEN_TIME] = self._section_greentime
        self.dataDic[SECTION_RESULT.VOLUME] = self.__section_volumes
        self.dataDic[SECTION_RESULT.SPEED_INT] = self.__section_speedint
        self.__define_direction()

    def __define_direction(self):
        if not hasattr(self, 'direction') or self.direction is None:
            self.direction = None if len(self.stations) == 0 else self.stations[0].direction

    def addStation(self, station):
        self.stations.append(station)
        self.__define_direction()

    def collect_data(self):
        if len(self.__section_volumes) == 0:
            return 0, 0, 0, 0
        else:
            return self.__section_co2[-1], self.__section_volumes[-1], self.__section_queues[-1], self._section_greentime[-1]

    def getCurrentQueue(self):
        if len(self.__section_queues) > 0:
            return self.__section_queues[-1]
        else:
            return 0

    def getCurrentCO2(self):
        if len(self.__section_co2) > 0:
            return self.__section_co2[-1]
        else:
            return 0

    def getCurrentVol(self):
        if len(self.__section_volumes) > 0:
            return self.__section_volumes[-1]
        else:
            return 0

    def getCurrentGreenTime(self):
        if len(self._section_greentime) > 0:
            return self._section_greentime[-1]
        else:
            return 0

    def getCurrentTime(self):
        if len(self.__time) > 0:
            return self.__time[-1]
        return 0

    def getDatabyID(self, id: SECTION_RESULT):
        return self.dataDic[id]

    def update(self, time):
        pass

    def print(self):
        print('this is Section!!')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['direction']  # direction은 계산 가능한 필드이므로 직렬화에서 제외
        state['__class__'] = Section
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # direction은 stations의 첫 번째 항목으로부터 다시 계산하여 설정
        self.__define_direction()
        self.__class__ = DSection#state.pop('__class__', Section)

class SSection(Section):
    def __init__(self, id, stations=None):
        super().__init__(id, stations)

        #for data
        self.traffic_queue = 0
        self.current_greentime = -1
        self.section_vehicles = set()

    def setGreenTime(self, greetime, logic):
        self.current_greentime = greetime
        self.__setSignalGreenTime(greetime, logic)

    def updateGreentime(self):
        if self.current_greentime != -1:
            self.append_section_greentime(self.current_greentime)
        else:
            if len(self._section_greentime) > 0:
                self.append_section_greentime(self._section_greentime[-1])
            else:
                self.append_section_greentime(self.default_greentime)


    def __setSignalGreenTime(self, time, logic):
        logic.phases[self.direction.value[1]].duration = time

    def update(self, time):
        section_co2_emission = 0
        section_volume = 0
        removal_veh = list()
        speedcnt = 0
        pscnt = len(self.getDatabyID(SECTION_RESULT.SPEED_INT))
        isspeedadded = False
        speedsum = 0
        for i, station in enumerate(self.stations):
            #update station data
            station.update()

            if i == 0:
                section_volume += station.getVolume()
                self.section_vehicles.update(station.getInputVehIds())

            #update input station data according to InputStation Setup
            if station.id == get_input_station_value(self.direction):
                self.traffic_queue += station.getVolume()

            self.traffic_queue -= station.getExitVolume()
            self.section_vehicles.difference_update(station.getExitVehIds())

            scnt = len(station.getSpeedInts())

            if i == 0 and scnt > pscnt:
                isspeedadded = True
                sspeed = station.getSpeedInt()
                if sspeed != -1 :
                    speedsum += station.getSpeedInt()
                    speedcnt += 1
            # if self.id == '2':
            #     if station.getExitVolume() > 0:
            #         print('----exit vol : ',station.getExitVolume())

        #calculate average speed
        if isspeedadded is True:
            average_speed_int = speedsum / speedcnt * SMUtil.MPStoKPH
            self.append_section_speedint(average_speed_int)
            self.append_section_timeint(time)

        for vehicle in self.section_vehicles:
            try:
                if traci.vehicle.getCO2Emission(vehicle) >= 0:
                    section_co2_emission += traci.vehicle.getCO2Emission(vehicle) / 1000
            except TraCIException:
                print('------------------------disappear -> ',vehicle)
                #self.section_vehicles.remove(vehicle)
                removal_veh.append(vehicle)

        self.section_vehicles.difference_update(removal_veh)

        self.append_section_queues(self.traffic_queue)
        self.append_section_co2(section_co2_emission)
        self.append_section_volumes(section_volume)
        self.append_section_time(time)
        self.updateGreentime()
        # if self.id == '2':
        #     print('Sid : ',self.id, ', Queue : ')
        #     print('---- speedint : ', self.getDatabyID(SECTION_RESULT.SPEED_INT))
        #self.collect_data()
    def print(self):
        print('this is SSection!!')

class DSection(Section):
    def __init__(self, id, stations=None):
        super().__init__(id, stations)

    def update(self):
        pass

    def print(self):
        print('this is DSection!!')

class Infra:
    def __init__(self, sumocfg_path, scenario_path, scenario_file, sections, sigtype=None):
        self.sumocfg_path = sumocfg_path
        # SUMO Scenario File Path
        self.scenario_path = scenario_path
        # SUMO Scenario File(.add.xml)
        self.scenario_file = scenario_file
        self.__sections = sections

        self.__time = deque()
        self.__totalCO2 = deque()
        self.__totalCO2ACC = deque()
        self.__totalVolume = deque()
        self.append_time = self.__time.append
        self.append_totalCO2 = self.__totalCO2.append
        self.append_totalCO2ACC = self.__totalCO2ACC.append
        self.append_totalVolume = self.__totalVolume.append
        self.dataDic = {}
        self.dataDic[TOTAL_RESULT.TIME] = self.__time
        self.dataDic[TOTAL_RESULT.TOTAL_CO2] = self.__totalCO2
        self.dataDic[TOTAL_RESULT.TOTAL_CO2_ACC] = self.__totalCO2ACC
        self.dataDic[TOTAL_RESULT.TOTAL_VOLUME] = self.__totalVolume

        self.sigType: str = sigtype
        self.__savedTime: datetime = None
        self.__savefileName: str = None

    def update(self):
        totalCO2 = 0
        totalCO2ACC = 0
        totalVol = 0
        time = traci.simulation.getTime()
        for section_id, section in self.getSections().items():
            section.update(time)
            totalCO2 += section.getCurrentCO2()
            totalVol += section.getCurrentVol()

        # vehicle_ids = traci.vehicle.getIDList()
        # for vehicle_id in vehicle_ids:
        #     totalCO2 += traci.vehicle.getCO2Emission(vehicle_id) / 1000

        totalCO2ACC = self.__totalCO2ACC[-1] + totalCO2 if len(self.__totalCO2ACC) > 0 else totalCO2
        totalVol = self.__totalVolume[-1] + totalVol if len(self.__totalVolume) > 0 else totalVol
        self.append_totalCO2(totalCO2)
        self.append_totalCO2ACC(totalCO2ACC)
        self.append_totalVolume(totalVol)
        self.append_time(time)

    def getDatabyID(self, totalresult: TOTAL_RESULT):
        return self.dataDic[totalresult]

    def getDatabyName(self, name: str):
        return self.getDatabyID(TOTAL_RESULT.from_string(name))

    def getTotalCO2(self):
        if len(self.__totalCO2) > 0:
            return self.__totalCO2[-1]
        else:
            return 0

    def getTotalCO2mg(self):
        return self.getTotalCO2() * 100

    def getTime(self):
        return self.__time

    def getSections(self) -> dict:
        return self.__sections

    def setSaveFileName(self, name=None):
        if self.__savedTime is None:
            self.setCurrentTime()

        filename = self.sigType + '_'
        if name is not None:
            filename = name + '_' + filename

        formatted_time = self.__savedTime.strftime("%Y%m%d%H%M%S")
        filename = filename+formatted_time

        self.__savefileName = filename + '.data'
        return self.__savefileName

    def getFileName(self):
        return self.__savefileName

    def getSavedTime(self):
        return self.__savedTime

    def setCurrentTime(self):
        self.__savedTime = datetime.now()
        print(self.__savedTime)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    # 역직렬화된 데이터를 객체 상태에 복원하는 메서드
    def __setstate__(self, state):
        self.__dict__.update(state)
