import os

def split_obj2part(objfile):
    edge_map = {}
    point_map = {}
    big_obj = []
    with open(objfile, "r") as fil:
        k = 0
        while True:
            k += 1
            line = fil.readline()
            if not line:
                break

            if line[0] == "v":
                point_lst = line.split()
                pointline = [float(point_lst[1]), float(point_lst[2]), float(point_lst[3]), point_lst[4]]
                big_obj.append(pointline)

            elif line[0] == "f":
                edge_lst = line.split()
                if big_obj[int(edge_lst[1])-1][3] != big_obj[int(edge_lst[2])-1][3]:
                    for f in range(1, 3):
                        first_label = big_obj[int(edge_lst[f])-1][3]
                        edge = []
                        for i in range(1, 4):

                            point = big_obj[int(edge_lst[i])-1][0:3]

                            if first_label not in edge_map:
                                edge_map[first_label] = []
                            if first_label not in point_map:
                                point_map[first_label] = []

                            if point not in point_map[first_label]:
                                point_map[first_label].append(point)

                            edge.append(point_map[first_label].index(point)+1)

                        edge_map[first_label].append(edge)

                elif big_obj[int(edge_lst[1]) - 1][3] != big_obj[int(edge_lst[3]) - 1][3]:
                    for f in [1, 3]:
                        first_label = big_obj[int(edge_lst[f]) - 1][3]
                        edge = []
                        for i in range(1, 4):

                            point = big_obj[int(edge_lst[i]) - 1][0:3]

                            if first_label not in edge_map:
                                edge_map[first_label] = []
                            if first_label not in point_map:
                                point_map[first_label] = []

                            if point not in point_map[first_label]:
                                point_map[first_label].append(point)

                            edge.append(point_map[first_label].index(point) + 1)

                        edge_map[first_label].append(edge)

                elif big_obj[int(edge_lst[1]) - 1][3] != big_obj[int(edge_lst[2]) - 1][3] != big_obj[int(edge_lst[3]) - 1][3]:
                    for f in range(1, 4):
                        first_label = big_obj[int(edge_lst[f]) - 1][3]
                        edge = []
                        for i in range(1, 4):

                            point = big_obj[int(edge_lst[i]) - 1][0:3]

                            if first_label not in edge_map:
                                edge_map[first_label] = []
                            if first_label not in point_map:
                                point_map[first_label] = []

                            if point not in point_map[first_label]:
                                point_map[first_label].append(point)

                            edge.append(point_map[first_label].index(point) + 1)
                else:
                        first_label = big_obj[int(edge_lst[1]) - 1][3]
                        edge = []
                        for i in range(1, 4):

                            point = big_obj[int(edge_lst[i]) - 1][0:3]

                            if first_label not in edge_map:
                                edge_map[first_label] = []
                            if first_label not in point_map:
                                point_map[first_label] = []

                            if point not in point_map[first_label]:
                                point_map[first_label].append(point)

                            edge.append(point_map[first_label].index(point) + 1)

                        edge_map[first_label].append(edge)
    fil.close()

    return point_map, edge_map


def main(file_path, part_savedir):

    num2affordance = {"0": "no_affordance", "1": 'handle-grasp', "3": 'press', "4": 'lift',
                      "5": 'wrap-grasp', "6": 'Twist', "7": 'Support', "8": 'Pull', "9": 'Lever'}

    point_map_new, edge_map_new = split_obj2part(file_path)
    os.makedirs(part_savedir, exist_ok=True)

    for id, points_list in point_map_new.items():

        with open(os.path.join(part_savedir, "part_{}.obj".format(num2affordance[id])), "w") as p:

            for new_point_list in points_list:
                new_line = "v" + " " + str(new_point_list[0]) + " " + str(new_point_list[1]) + " " + str(new_point_list[2]) + " " + "{}\n".format(id)
                p.writelines(new_line)

            for edge_list in edge_map_new[id]:
                new_line = "f" + " " + str(edge_list[0]) + " " + str(edge_list[1]) + " " + str(edge_list[2]) + "\n"
                p.writelines(new_line)
        p.close()
    print("Split object to part finished: {}".format(file_path))

if __name__ == "__main__":
    object_path = r"./AffordPose/samples/bottle_mesh.obj"
    part_savedir = r"./AffordPose/samples/part_obj"
    main(object_path, part_savedir)
