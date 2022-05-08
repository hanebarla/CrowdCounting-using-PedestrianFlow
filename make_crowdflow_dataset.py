import random
import csv
from progress.bar import Bar
import argparse


def main():
    """
    csv format
    --------
        train
            index 0: input image(step t),
            index 1: person label(step t),
            index 2: input label(step t-1),
            index 3: person label(step t-1),
            index 4: label flow(step t-1 2 t),
            index 5: input image(step t+1),
            index 6: preson label(step t+1),
            index 7: label flow(step t 2 t+1)

        test
            index 0: input image(step tm),
            index 1: person label(step tm),
            index 2: input image(step t),
            index 3: person label(step t),
            index 4: label flow(step tm 2 t)
    """
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the root folder of the Datasets.
                                                 In default, path is 'E:/Dataset/TUBCrowdFlow/'
                                                 """)
    parser.add_argument('-p', '--path', default='E:/Dataset/TUBCrowdFlow/')
    args = parser.parse_args()

    AllPathList = []
    AllPathDict = {}
    TrainPathList = []
    TrainPathDict = {}
    TestPathList = []
    TestPathDict = {}

    frame_num_list = [300, 300, 250, 300, 450]
    DatasetFolder = args.path
    ImgFolder = DatasetFolder + "images/"
    GTTrajFolder = DatasetFolder + "gt_trajectories/"
    GTFlowFolder = DatasetFolder + "gt_flow/"
    GTPersonFolder = "PersonTrajectories/"
    SceneFolderNameLis = [
        "IM01/", "IM01_hDyn/",
        "IM02/", "IM02_hDyn/",
        "IM03/", "IM03_hDyn/",
        "IM04/", "IM04_hDyn/",
        "IM05/", "IM05_hDyn/"
    ]

    bar = Bar('Makeing csv... ', max=len(SceneFolderNameLis))
    for i, scene in enumerate(SceneFolderNameLis):
        frame_num = frame_num_list[int(i / 2)]
        gtTraj_img_path = GTTrajFolder + scene + GTPersonFolder

        tmpPathList = []
        tmpPathDict = {}

        for fr in range(frame_num - 2):
            tm = fr
            t = fr + 1
            tp = fr + 2

            t_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(t)
            tm_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(tm)
            tp_img_path = ImgFolder + scene + "frame_{:0=4}.png".format(tp)

            t_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(t)
            tm_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(tm)
            tp_person_img_path = gtTraj_img_path + "PersonTrajectories_frame_{:0=4}.png".format(tp)

            tm2t_flow_path = GTFlowFolder + scene + "frameGT_{:0=4}.png".format(tm)
            t2tp_flow_path = GTFlowFolder + scene + "frameGT_{:0=4}.png".format(t)

            PathList_per_frame = [
                t_img_path, t_person_img_path,
                tm_img_path, tm_person_img_path, tm2t_flow_path,
                tp_img_path, tp_person_img_path, t2tp_flow_path
            ]

            tmpPathList.append(t_img_path)
            tmpPathDict[t_img_path] = PathList_per_frame

            if fr == 0:
                all_pathlist_per_frame = [tm_img_path, tm_person_img_path,
                                          t_img_path, t_person_img_path,
                                          tm2t_flow_path]
                AllPathList.append(tm_img_path)
                AllPathDict[tm_img_path] = all_pathlist_per_frame

            all_pathlist_per_frame = [t_img_path, t_person_img_path,
                                      tp_img_path, tp_person_img_path,
                                      t2tp_flow_path]
            AllPathList.append(t_img_path)
            AllPathDict[t_img_path] = all_pathlist_per_frame

            if int(i / 2) < 3:  # TrainPathes
                PathList_per_frame = [t_img_path, t_person_img_path,
                                      tm_img_path, tm_person_img_path, tm2t_flow_path,
                                      tp_img_path, tp_person_img_path, t2tp_flow_path]
                TrainPathList.append(t_img_path)
                TrainPathDict[t_img_path] = PathList_per_frame
            else:  # TestPathes
                if fr == 0:
                    tm_PathList_per_frame = [tm_img_path, tm_person_img_path,
                                             t_img_path, t_person_img_path,
                                             tm2t_flow_path]
                    TestPathList.append(tm_img_path)
                    TestPathDict[tm_img_path] = tm_PathList_per_frame

                t_PathList_per_frame = [t_img_path, t_person_img_path,
                                        tp_img_path, tp_person_img_path,
                                        t2tp_flow_path]
                TestPathList.append(t_img_path)
                TestPathDict[t_img_path] = t_PathList_per_frame

        with open("Scene_{}.csv".format(scene.replace("/", "")), "w", newline='') as f:
            writer = csv.writer(f)
            for path in tmpPathList:
                writer.writerow(tmpPathDict[path])

        bar.next()
    bar.finish()

    random.shuffle(TrainPathList)
    with open("TrainData_Path.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for path in TrainPathList:
            writer.writerow(TrainPathDict[path])

    with open("TestData_Path.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for path in TestPathList:
            writer.writerow(TestPathDict[path])

    with open("AllData_Path.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for path in AllPathList:
            writer.writerow(AllPathDict[path])

    print("Done")


if __name__ == "__main__":
    main()
